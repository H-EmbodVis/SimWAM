"""SimWAM trajectory prediction over the NAVSIM v2 ``navhard_two_stage`` split.

For every navhard token (original stage-one **and** synthetic stage-two scenes) this script
runs SimWAM action inference and writes the predicted 4s@2Hz trajectory to ``<token>.npy``
(shape ``[num_poses, 3]`` = ``x, y, heading`` in the current ego / rear-axle frame). The npy
directory is then consumed by ``navsim_v2``'s ``NpyTrajectoryAgent`` under the two-stage
``run_pdm_score.py`` to produce the pseudo-closed-loop EPDMS.

Why a dedicated script (vs ``eval_navsim.py``): navhard is a v2 two-stage dataset with synthetic
scenes; SimWAM's ``NavSimVideoDataset`` uses the v1 ``navsim.SceneLoader`` (no synthetic
support). Here we drive the **v2** ``SceneLoader`` directly (``import navsim`` already resolves to
navsim_v2 in this environment) and reuse SimWAM's model + the dataset's preprocessing/prompt/
denorm helpers so results match training exactly.

Launch with torchrun (tokens are sharded across ranks); see ``run_predict_navhard.sh``.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as transforms_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
for _p in (str(project_root / "navsim_v2"), str(project_root), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse model/checkpoint/distributed helpers from the existing eval script (same directory).
from eval_navsim import (  # noqa: E402
    _get_local_rank,
    _get_rank,
    _get_world_size,
    _init_distributed,
    _is_main_process,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_eval_device,
)
from simwam.datasets.navsim.navsim_dataset import NavSimVideoDataset  # noqa: E402

# navsim MUST resolve to navsim_v2 (synthetic-scene support). Fail fast otherwise.
import navsim  # noqa: E402
from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

logger = logging.getLogger(__name__)


def _assert_navsim_v2() -> None:
    navsim_file = str(getattr(navsim, "__file__", ""))
    if "navsim_v2" not in navsim_file:
        raise RuntimeError(
            "predict_navhard requires the navsim_v2 package (synthetic-scene support), but "
            f"`import navsim` resolved to: {navsim_file}. Ensure navsim_v2 is the active install "
            "(pip install -e navsim_v2) or prepend it to PYTHONPATH."
        )


def _default(eval_cfg: DictConfig, key: str, value: Any) -> None:
    if key not in eval_cfg or eval_cfg.get(key) is None:
        eval_cfg[key] = value


def _ensure_eval_defaults(cfg: DictConfig) -> None:
    if "EVALUATION" not in cfg or cfg.EVALUATION is None:
        cfg.EVALUATION = OmegaConf.create({})
    eval_cfg = cfg.EVALUATION
    openscene_root = os.environ.get("OPENSCENE_DATA_ROOT", "./data/navsim")
    scene_filter_default = "./navsim_v2/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"
    defaults = {
        "openscene_data_root": openscene_root,
        "navsim_log_path": None,          # default: <root>/navsim_logs/test
        "original_sensor_path": None,     # default: <root>/sensor_blobs/test
        "synthetic_sensor_path": None,    # default: <root>/navhard_two_stage/sensor_blobs
        "synthetic_scenes_path": None,    # default: <root>/navhard_two_stage/synthetic_scene_pickles
        "scene_filter": scene_filter_default,
        "action_output_dir": None,        # default: evaluate_results_v2/navhard/<exp>/pred_actions
        "max_tokens": None,               # smoke-test cap
        "num_inference_steps": None,      # default: cfg.eval_num_inference_steps or 20
        "seed": 0,
        "device": None,
        "overwrite": False,               # skip tokens whose npy already exists (resume)
    }
    for key, value in defaults.items():
        _default(eval_cfg, key, value)

    if eval_cfg.action_output_dir is None:
        exp = str(cfg.get("experiment_name", "simwam_navhard"))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        eval_cfg.action_output_dir = str(
            Path("evaluate_results_v2") / "navhard" / exp / timestamp / "pred_actions"
        )


def _resolve_paths(eval_cfg: DictConfig) -> dict[str, Path]:
    root = Path(os.path.expanduser(os.path.expandvars(str(eval_cfg.openscene_data_root))))

    def _pick(key: str, default: Path) -> Path:
        val = eval_cfg.get(key)
        return Path(os.path.expanduser(os.path.expandvars(str(val)))) if val is not None else default

    return {
        "navsim_log_path": _pick("navsim_log_path", root / "navsim_logs" / "test"),
        "original_sensor_path": _pick("original_sensor_path", root / "sensor_blobs" / "test"),
        "synthetic_sensor_path": _pick("synthetic_sensor_path", root / "navhard_two_stage" / "sensor_blobs"),
        "synthetic_scenes_path": _pick("synthetic_scenes_path", root / "navhard_two_stage" / "synthetic_scene_pickles"),
    }


def _build_scene_loader(cfg: DictConfig) -> SceneLoader:
    eval_cfg = cfg.EVALUATION
    paths = _resolve_paths(eval_cfg)
    scene_filter = instantiate(OmegaConf.load(str(eval_cfg.scene_filter)))
    if not bool(getattr(scene_filter, "include_synthetic_scenes", False)):
        logger.warning(
            "scene_filter.include_synthetic_scenes is False; only original (stage-one) scenes "
            "will be predicted — the two-stage EPDMS needs synthetic scenes too."
        )
    # SimWAM navsim task uses the front camera only.
    sensor_config = SensorConfig(
        cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
        cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False,
    )
    return SceneLoader(
        data_path=paths["navsim_log_path"],
        original_sensor_path=paths["original_sensor_path"],
        scene_filter=scene_filter,
        synthetic_sensor_path=paths["synthetic_sensor_path"],
        synthetic_scenes_path=paths["synthetic_scenes_path"],
        sensor_config=sensor_config,
    )


def _preprocess_front_image(image_hw3: np.ndarray, video_size_hw: list[int]) -> torch.Tensor:
    """Mirror NavSimVideoDataset._build_video_tensor for a single front frame -> [1,3,H,W] in [-1,1]."""
    pil = Image.fromarray(np.asarray(image_hw3).astype(np.uint8))
    tensor = transforms_F.to_tensor(pil)  # [3,H,W] in [0,1]
    tensor = transforms_F.resize(
        tensor,
        size=list(video_size_hw),
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = transforms_F.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1,1]
    return tensor.unsqueeze(0).contiguous()  # [1,3,H,W]


def _load_text_context(cache_dir: Optional[str], context_len: int, prompt: str):
    """Replicates NavSimVideoDataset._get_cached_text_context (read-only, precomputed cache)."""
    if cache_dir is None:
        raise ValueError("`text_embedding_cache_dir` is required (SimWAM expects precomputed context).")
    import hashlib

    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = Path(cache_dir) / f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing prompt cache: {cache_path}. Precompute it with "
            "scripts/precompute_navsim_text_embeds.py before running."
        )
    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"]
    context_mask = payload["mask"].bool()
    if context.shape[0] != context_len or context_mask.shape[0] != context_len:
        raise ValueError(f"Cached prompt length mismatch: expected {context_len}, got {context.shape[0]}")
    context = context.to(dtype=torch.bfloat16)
    context[~context_mask] = 0
    context_mask = torch.ones_like(context_mask)
    return context.contiguous(), context_mask.contiguous()


def _denormalize_action(pred_norm: torch.Tensor, trajectory_mode: str, normalize_action: bool) -> torch.Tensor:
    """Mirror NavSimVideoDataset.denormalize_action for a single [T,3] (or [1,T,3]) prediction."""
    a = pred_norm.detach().to(device="cpu", dtype=torch.float32)
    if a.ndim == 3:
        a = a[0]
    if trajectory_mode == "relative":
        if normalize_action:
            a = NavSimVideoDataset.denorm_odo_relative(a)
        return NavSimVideoDataset._relative_deltas_to_absolute(a)
    if not normalize_action:
        return a
    return NavSimVideoDataset.denorm_odo(a)


def _build_prompt_for_scene(scene, num_history_frames: int, use_dynamic_prompt: bool) -> str:
    ego = scene.get_agent_input().ego_statuses[-1]
    velocity = torch.tensor(ego.ego_velocity, dtype=torch.float32)
    acceleration = torch.tensor(ego.ego_acceleration, dtype=torch.float32)
    driving_command = torch.tensor(ego.driving_command, dtype=torch.float32)
    speed_mps = float(torch.linalg.norm(velocity).item())
    acc_mps2 = float(torch.linalg.norm(acceleration).item())
    if use_dynamic_prompt:
        hist_xyh = torch.tensor(
            scene.get_history_trajectory(num_trajectory_frames=num_history_frames).poses,
            dtype=torch.float32,
        )
    else:
        hist_xyh = torch.zeros((1, 3), dtype=torch.float32)  # ignored by the fixed prompt
    return NavSimVideoDataset.build_prompt_fixed(
        hist_xyh=hist_xyh,
        high_cmd_one_hot=driving_command,
        speed_mps=speed_mps,
        acc_mps2=acc_mps2,
        use_dynamic_prompt=use_dynamic_prompt,
    )


def _extract_proprio(scene) -> torch.Tensor:
    ego = scene.get_agent_input().ego_statuses[-1]
    velocity = torch.tensor(ego.ego_velocity, dtype=torch.float32)
    acceleration = torch.tensor(ego.ego_acceleration, dtype=torch.float32)
    driving_command = torch.tensor(ego.driving_command, dtype=torch.float32)
    return torch.cat([velocity, acceleration, driving_command], dim=0)  # [8]


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
    num_inference_steps = int(cfg.get("eval_num_inference_steps", 20)) if num_inference_steps is None else int(num_inference_steps)
    seed = None if eval_cfg.get("seed") is None else int(eval_cfg.get("seed"))
    overwrite = bool(eval_cfg.get("overwrite", False))

    rank, world_size = _get_rank(), _get_world_size()
    tokens = list(loader.tokens)
    max_tokens = eval_cfg.get("max_tokens", None)
    if max_tokens is not None:
        tokens = tokens[: int(max_tokens)]
    synthetic_tokens = set(loader.synthetic_scenes_tokens)
    my_tokens = tokens[rank::world_size]

    # Text context is cached per unique prompt (one entry when use_dynamic_prompt is False).
    context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    n_ok = n_fail = n_skip = n_syn = 0
    progress = tqdm(my_tokens, desc=f"predict navhard rank{rank}", disable=not _is_main_process())
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
            if token in synthetic_tokens:
                n_syn += 1
        except Exception:
            logger.warning("Prediction failed for token %s", token, exc_info=True)
            n_fail += 1

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return {"rank": rank, "n_ok": n_ok, "n_syn": n_syn, "n_fail": n_fail, "n_skip": n_skip, "n_local": len(my_tokens)}


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_navsim.yaml")
def main(cfg: DictConfig) -> None:
    _assert_navsim_v2()
    OmegaConf.set_struct(cfg, False)  # allow injecting navhard-specific EVALUATION keys in code
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
            logging.FileHandler(out_dir.parent / (f"predict.rank{rank:02d}.log" if not _is_main_process() else "predict.log"), encoding="utf-8"),
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
        logger.info("navhard tokens: total=%d original=%d synthetic=%d",
                    len(loader.tokens), len(loader.tokens_stage_one), len(loader.synthetic_scenes_tokens))
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
