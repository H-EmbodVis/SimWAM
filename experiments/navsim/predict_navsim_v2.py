"""SimWAM trajectory prediction under NAVSIM **v2**, trajectory-only (no metrics).

For every token of a v2 scene-filter split this runs SimWAM action inference and writes the
predicted 4s@2Hz trajectory to ``<token>.npy`` (shape ``[num_poses, 3]`` = ``x, y, heading`` in
the current ego / rear-axle frame). The npy dir is then consumed by ``navsim_v2``'s
``NpyTrajectoryAgent`` under ``run_pdm_score.py`` to compute EPDMS (that scoring is a separate
step; this script does NOT compute any metric).

Relationship to the other scripts:
  * ``eval_navsim.py``      -> v1 devkit, predicts AND scores PDMS (navtest).
  * ``predict_navhard.py``  -> v2, predict-only, **hard-wired to navhard_two_stage** (synthetic scenes).
  * ``predict_navsim_v2.py`` (this) -> v2, predict-only, **general**: any v2 scene-filter split,
    synthetic scenes optional. Default target is single-stage ``navtest``.

Why a dedicated v2 script (vs eval_navsim.py): SimWAM's ``NavSimVideoDataset`` uses the v1
``navsim.SceneLoader``. Here we drive the **v2** ``SceneLoader`` directly (``import navsim`` must
resolve to navsim_v2) and reuse the exact SimWAM inference helpers so results match training.

Launch with torchrun (tokens are sharded across ranks); see ``run_predict_navsim_v2.sh``.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
for _p in (str(project_root / "navsim_v2"), str(project_root), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Model / checkpoint / distributed helpers (shared with the v1 eval).
from eval_navsim import (  # noqa: E402
    _get_rank,
    _get_world_size,
    _init_distributed,
    _is_main_process,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_eval_device,
)

# Generic v2 per-scene inference helpers (identical to what navhard prediction uses -- reuse, don't
# duplicate). These are NOT navhard-specific: image preprocess, text-context cache, action denorm,
# prompt build, proprio extraction, and the navsim_v2 guard.
from predict_navhard import (  # noqa: E402
    _assert_navsim_v2,
    _build_prompt_for_scene,
    _default,
    _denormalize_action,
    _extract_proprio,
    _load_text_context,
    _preprocess_front_image,
)

import navsim  # noqa: E402  (must resolve to navsim_v2; enforced by _assert_navsim_v2)
from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

logger = logging.getLogger(__name__)

# Default v2 scene filter: single-stage navtest (no synthetic scenes required).
_DEFAULT_SCENE_FILTER = "./navsim_v2/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"


def _ensure_eval_defaults(cfg: DictConfig) -> None:
    if "EVALUATION" not in cfg or cfg.EVALUATION is None:
        cfg.EVALUATION = OmegaConf.create({})
    eval_cfg = cfg.EVALUATION
    openscene_root = os.environ.get("OPENSCENE_DATA_ROOT", "./data/navsim")
    defaults = {
        "openscene_data_root": openscene_root,
        # Paths default to the standard NAVSIM env vars (as the shell scripts set them), else the
        # trainval blobs used by navtest. Override per split via EVALUATION.* or env.
        "navsim_log_path": os.environ.get("NAVSIM_LOG_PATH"),
        "original_sensor_path": os.environ.get("NAVSIM_SENSOR_BLOBS_PATH"),
        # Synthetic scenes: only needed for two-stage splits (navhard/navtest_two_stage/...).
        "synthetic_sensor_path": None,
        "synthetic_scenes_path": None,
        "scene_filter": _DEFAULT_SCENE_FILTER,
        "action_output_dir": None,        # default: evaluate_results_v2/<filter>/<exp>/<ts>/pred_actions
        "max_tokens": None,               # smoke-test cap
        "num_inference_steps": None,      # default: cfg.eval_num_inference_steps or 20
        "seed": 0,
        "device": None,
        "overwrite": False,               # skip tokens whose npy already exists (resume)
    }
    for key, value in defaults.items():
        _default(eval_cfg, key, value)

    if eval_cfg.action_output_dir is None:
        split_name = Path(str(eval_cfg.scene_filter)).stem
        exp = str(cfg.get("experiment_name", "simwam_v2"))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        eval_cfg.action_output_dir = str(
            Path("evaluate_results_v2") / split_name / exp / timestamp / "pred_actions"
        )


def _resolve_path(eval_cfg: DictConfig, key: str, default: Path) -> Path:
    val = eval_cfg.get(key)
    return Path(os.path.expanduser(os.path.expandvars(str(val)))) if val is not None else default


def _build_scene_loader(cfg: DictConfig) -> SceneLoader:
    eval_cfg = cfg.EVALUATION
    root = Path(os.path.expanduser(os.path.expandvars(str(eval_cfg.openscene_data_root))))
    navsim_log_path = _resolve_path(eval_cfg, "navsim_log_path", root / "navsim_logs" / "trainval")
    original_sensor_path = _resolve_path(eval_cfg, "original_sensor_path", root / "sensor_blobs" / "trainval")

    scene_filter = instantiate(OmegaConf.load(str(eval_cfg.scene_filter)))
    needs_synthetic = bool(getattr(scene_filter, "include_synthetic_scenes", False))

    synthetic_sensor_path = None
    synthetic_scenes_path = None
    if needs_synthetic:
        synthetic_sensor_path = _resolve_path(
            eval_cfg, "synthetic_sensor_path", root / "navhard_two_stage" / "sensor_blobs"
        )
        synthetic_scenes_path = _resolve_path(
            eval_cfg, "synthetic_scenes_path", root / "navhard_two_stage" / "synthetic_scene_pickles"
        )

    # SimWAM navsim task uses the front camera only.
    sensor_config = SensorConfig(
        cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
        cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False,
    )
    return SceneLoader(
        data_path=navsim_log_path,
        original_sensor_path=original_sensor_path,
        scene_filter=scene_filter,
        synthetic_sensor_path=synthetic_sensor_path,
        synthetic_scenes_path=synthetic_scenes_path,
        sensor_config=sensor_config,
    )


@torch.no_grad()
def predict(cfg: DictConfig, model: torch.nn.Module, loader: SceneLoader, out_dir: Path) -> dict[str, Any]:
    eval_cfg = cfg.EVALUATION
    data_cfg = cfg.data.train
    trajectory_mode = str(data_cfg.get("trajectory_mode", "absolute"))
    normalize_action = bool(data_cfg.get("normalize_action", True))
    use_dynamic_prompt = bool(data_cfg.get("use_dynamic_prompt", False))
    video_size = list(data_cfg.get("video_size", [384, 672]))
    future_action_horizon = int(data_cfg.get("future_action_horizon", 8))
    context_len = int(data_cfg.get("context_len", 256))
    text_cache_dir = data_cfg.get("text_embedding_cache_dir", None)
    text_cache_dir = None if text_cache_dir is None else str(text_cache_dir)

    num_inference_steps = eval_cfg.get("num_inference_steps", None)
    num_inference_steps = (
        int(cfg.get("eval_num_inference_steps", 20)) if num_inference_steps is None else int(num_inference_steps)
    )
    seed = None if eval_cfg.get("seed") is None else int(eval_cfg.get("seed"))
    overwrite = bool(eval_cfg.get("overwrite", False))

    rank, world_size = _get_rank(), _get_world_size()
    tokens = list(loader.tokens)
    max_tokens = eval_cfg.get("max_tokens", None)
    if max_tokens is not None:
        tokens = tokens[: int(max_tokens)]
    my_tokens = tokens[rank::world_size]

    # Text context is cached per unique prompt (one entry when use_dynamic_prompt is False).
    context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    n_ok = n_fail = n_skip = 0
    progress = tqdm(my_tokens, desc=f"predict v2 rank{rank}", disable=not _is_main_process())
    for token in progress:
        out_path = out_dir / f"{token}.npy"
        if not overwrite and out_path.exists():
            n_skip += 1
            continue
        try:
            scene = loader.get_scene_from_token(token)
            num_history_frames = int(scene.scene_metadata.num_history_frames)
            cur = max(0, min(num_history_frames - 1, len(scene.frames) - 1))
            image = scene.frames[cur].cameras.cam_f0.image
            if image is None:
                raise ValueError(f"cam_f0 image is None for token {token}")
            input_image = _preprocess_front_image(image, video_size)
            proprio = _extract_proprio(scene)

            prompt = _build_prompt_for_scene(scene, num_history_frames, use_dynamic_prompt)
            if prompt not in context_cache:
                context_cache[prompt] = _load_text_context(text_cache_dir, context_len, prompt)
            context, context_mask = context_cache[prompt]

            pred = model.infer_action(
                prompt=None,
                input_image=input_image,
                action_horizon=future_action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
            poses = _denormalize_action(pred["action"], trajectory_mode, normalize_action)
            poses_np = poses.numpy().astype(np.float32)
            if poses_np.shape != (future_action_horizon, 3):
                raise ValueError(f"unexpected pred shape {poses_np.shape} for token {token}")
            np.save(out_path, poses_np)
            n_ok += 1
        except Exception:
            logger.warning("Prediction failed for token %s", token, exc_info=True)
            n_fail += 1

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return {"rank": rank, "n_ok": n_ok, "n_fail": n_fail, "n_skip": n_skip, "n_local": len(my_tokens)}


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_navsim.yaml")
def main(cfg: DictConfig) -> None:
    _assert_navsim_v2()
    OmegaConf.set_struct(cfg, False)  # allow injecting v2-specific EVALUATION keys in code
    _ensure_eval_defaults(cfg)
    _init_distributed()

    out_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.action_output_dir))))
    out_dir.mkdir(parents=True, exist_ok=True)

    rank = _get_rank()
    logging.basicConfig(
        level=logging.INFO if _is_main_process() else logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                out_dir.parent / (f"predict.rank{rank:02d}.log" if not _is_main_process() else "predict.log"),
                encoding="utf-8",
            ),
        ],
        force=True,
    )

    if cfg.get("ckpt") is None:
        raise ValueError("ckpt must not be None. Pass ckpt=./weights/checkpoint.pt")

    start = time.time()
    device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(device).eval()

    loader = _build_scene_loader(cfg)
    if _is_main_process():
        n_synth = len(getattr(loader, "synthetic_scenes_tokens", []) or [])
        logger.info(
            "v2 split=%s tokens: total=%d (synthetic=%d)",
            Path(str(cfg.EVALUATION.scene_filter)).stem, len(loader.tokens), n_synth,
        )
        logger.info("Output npy dir: %s", out_dir)
        logger.info("device=%s dtype=%s world_size=%d", device, model_dtype, _get_world_size())

    stats = predict(cfg, model, loader, out_dir)
    logger.info("rank%d done: %s", rank, stats)

    if _is_main_process():
        n_saved = len(list(out_dir.glob("*.npy")))
        logger.info("Saved %d/%d npy files to %s (%.1fs)", n_saved, len(loader.tokens), out_dir, time.time() - start)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
