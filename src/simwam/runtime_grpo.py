"""Runtime entry for SimWAM action-expert GRPO training.

Mirrors `simwam.runtime` but builds a `SimWAMGRPO`, warm-starts it from an IL
checkpoint, configures the GRPO sampler/objective, wires the NavSim PDM reward, and runs
`SimWAMGRPOTrainer`.
"""

import logging
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision, _resolve_train_device
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
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
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
    )


def run_grpo_training(cfg: DictConfig):
    misc.register_work_dir(cfg.output_dir)
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
        log_file=Path(cfg.output_dir) / "train_grpo.log",
    )
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), f)

    device = _resolve_train_device()
    model_dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision))

    # checkpoint_path is consumed here, not by the model factory -> strip before instantiate.
    checkpoint_path = cfg.model.get("checkpoint_path", None)
    # Resolve interpolations (e.g. proprio_dim) against the full cfg, then drop checkpoint_path.
    model_container = OmegaConf.to_container(cfg.model, resolve=True)
    model_container.pop("checkpoint_path", None)
    model = instantiate(model_container, model_dtype=model_dtype, device=device)

    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"GRPO warm-start checkpoint not found: {checkpoint_path}")
        logger.info("Warm-starting GRPO from IL checkpoint: %s", checkpoint_path)
        model.load_checkpoint(str(ckpt), optimizer=None)
    else:
        logger.warning("No `model.checkpoint_path`; GRPO starts from the ActionDiT pretrained backbone only.")

    model.configure_grpo(cfg.grpo)

    train_ds = instantiate(cfg.data.train)
    reward = instantiate_reward(cfg.grpo)

    trainer = SimWAMGRPOTrainer(model=model, train_dataset=train_ds, reward=reward, cfg=cfg)
    trainer.train()


def instantiate_reward(grpo_cfg: DictConfig):
    """Build the NAVSIM PDM reward used by action-only FlowGRPO."""
    from .datasets.navsim.pdm_reward import NavSimPDMReward

    kwargs = OmegaConf.to_container(grpo_cfg.reward, resolve=True)
    return NavSimPDMReward(**kwargs)
