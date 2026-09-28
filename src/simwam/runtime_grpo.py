"""Runtime entry for SimWAM action GRPO training.

Mirrors `simwam.runtime` but builds a `SimWAMGRPO`, warm-starts it from an IL
checkpoint, selects the NAVSIM PDM reward family or the Waymo Rater Feedback
Score, and runs the corresponding trainer. Reward backends are imported lazily
so each path only pays for its own dependencies.
"""

import logging
from pathlib import Path
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from .runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
)
from .trainer_grpo import SimWAMGRPOTrainer
from .utils import misc
from .utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def create_simwam_grpo(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = False,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_attention_mask_mode: str = "isolated",
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Build a `SimWAMGRPO` (same arg surface as `runtime.create_simwam`)."""
    from .models.wan22.simwam_grpo import SimWAMGRPO

    def _as_dict(value, name, required=False, default=None):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            if required:
                raise ValueError(f"`{name}` is required for SimWAMGRPO.")
            value = {} if default is None else default
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
        return value

    video_dit_config = _as_dict(video_dit_config, "video_dit_config", required=True)
    action_dit_config = _as_dict(action_dit_config, "action_dit_config")
    video_scheduler = _as_dict(video_scheduler, "video_scheduler")
    action_scheduler = _as_dict(action_scheduler, "action_scheduler", required=True)
    loss = _as_dict(loss, "loss")
    required_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required_keys - set(action_scheduler.keys())
    if missing:
        raise ValueError(f"`action_scheduler` missing keys: {sorted(missing)}.")
    return SimWAMGRPO.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        action_prediction_type=str(action_scheduler.get("prediction_type", "velocity")),
        action_sigma_clamp_min=float(action_scheduler.get("sigma_clamp_min", 0.1)),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        mot_attention_mask_mode=str(mot_attention_mask_mode),
    )


def run_grpo_training(cfg: DictConfig):
    target = str(cfg.grpo.get("target", "action"))
    if target != "action":
        raise ValueError(f"grpo.target must be 'action', got {target!r}")
    is_waymo_rfs = str(cfg.grpo.reward.get("name", "pdm")) == "rfs"
    if is_waymo_rfs and not cfg.model.get("checkpoint_path"):
        raise ValueError("Waymo RFS GRPO requires model.checkpoint_path pointing to the IL weights")
    misc.register_work_dir(cfg.output_dir)
    setup_logging(
        log_level=logging.INFO,
        is_main_process=(
            torch.distributed.get_rank() == 0
            if torch.distributed.is_initialized()
            else True
        ),
        log_file=Path(cfg.output_dir) / "train_grpo.log",
    )
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), f)
    device = _resolve_train_device()
    model_dtype = _mixed_precision_to_model_dtype(
        _normalize_mixed_precision(cfg.mixed_precision)
    )
    checkpoint_path = cfg.model.get("checkpoint_path", None)
    model_container = OmegaConf.to_container(cfg.model, resolve=True)
    model_container.pop("checkpoint_path", None)
    if is_waymo_rfs:
        # Seed before LoRA initialization, as well as the trainer's rollout seeding.
        from .utils.pytorch_utils import set_global_seed

        set_global_seed(int(cfg.seed))
    model = instantiate(model_container, model_dtype=model_dtype, device=device)
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"GRPO warm-start checkpoint not found: {checkpoint_path}"
            )
        logger.info("Warm-starting GRPO from IL checkpoint: %s", checkpoint_path)
        if is_waymo_rfs:
            from .datasets.waymo.checkpoint import load_waymo_checkpoint

            load_waymo_checkpoint(model, ckpt)
        else:
            model.load_checkpoint(str(ckpt), optimizer=None)
    else:
        logger.warning(
            "No `model.checkpoint_path`; GRPO starts from the ActionDiT pretrained backbone only."
        )
    model.configure_grpo(cfg.grpo)
    train_ds = instantiate(cfg.data.train)
    reward = instantiate_reward(cfg.grpo, train_dataset=train_ds)
    trainer_class = SimWAMGRPOTrainer
    if is_waymo_rfs:
        from .trainer_waymo_grpo import WaymoGRPOTrainer

        trainer_class = WaymoGRPOTrainer
    trainer = trainer_class(
        model=model, train_dataset=train_ds, reward=reward, cfg=cfg
    )
    trainer.train()


def instantiate_reward(grpo_cfg: DictConfig, train_dataset=None):
    """Select the NAVSIM PDM family for actions, or Waymo RFS.

    Imports stay local so the NAVSIM devkit is only pulled in by the PDM path.
    """
    target = str(grpo_cfg.get("target", "action"))
    if target != "action":
        raise ValueError(f"grpo.target must be 'action', got {target!r}")
    if str(grpo_cfg.reward.get("name", "pdm")) == "rfs":
        from .datasets.waymo.rfs_reward import WaymoRFSReward

        if train_dataset is None:
            raise ValueError("RFS requires the Waymo dataset to resolve token-specific references")
        return WaymoRFSReward(train_dataset)
    return _build_action_reward(grpo_cfg)


def _build_action_reward(grpo_cfg: DictConfig):
    kwargs = OmegaConf.to_container(grpo_cfg.reward, resolve=True)
    name = kwargs.pop("name", "pdm")
    kwargs.pop("epdms", None)
    if name in ("pdm", "pdm_log", "pdm_span"):
        from .datasets.navsim.pdm_reward import NavSimLogPDMReward, NavSimPDMReward, NavSimSpanPDMReward

        reward_cls = {"pdm": NavSimPDMReward, "pdm_log": NavSimLogPDMReward,
                      "pdm_span": NavSimSpanPDMReward}[name]
        return reward_cls(**kwargs)
    raise ValueError("grpo.reward.name must be 'pdm', 'pdm_log', 'pdm_span' or 'rfs', "
                     f"got {name!r}")
