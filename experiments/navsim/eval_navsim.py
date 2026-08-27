import gc
import inspect
import json
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
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _prefer_navsim_devkit() -> None:
    """Prefer the repository-relative NAVSIM devkit selected by the launcher."""
    devkit_root = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", project_root / "navsim"))
    if (devkit_root / "navsim" / "__init__.py").exists():
        sys.path.insert(0, str(devkit_root))


from simwam.utils.video_io import save_mp4
from simwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_cfg = cfg.EVALUATION
    explicit = eval_cfg.get("device", None)
    local_rank = _get_local_rank()
    if explicit is not None:
        explicit_str = str(explicit)
        if explicit_str == "cuda" and local_rank is not None and torch.cuda.is_available():
            return f"cuda:{local_rank}"
        return explicit_str
    if local_rank is not None and torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1"))


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> Optional[int]:
    if "LOCAL_RANK" not in os.environ:
        return None
    return int(os.environ["LOCAL_RANK"])


def _is_main_process() -> bool:
    return _get_rank() == 0


def _init_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available, but WORLD_SIZE > 1.")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    local_rank = _get_local_rank()
    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    payload = torch.load(ckpt, map_location="cpu")

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    if "mot" in payload and hasattr(model, "mot"):
        incompatible = model.mot.load_state_dict(payload["mot"], strict=False)
        missing_keys.extend(list(incompatible.missing_keys))
        unexpected_keys.extend(list(incompatible.unexpected_keys))
    elif "dit" in payload and hasattr(model, "dit"):
        incompatible = model.dit.load_state_dict(payload["dit"], strict=False)
        missing_keys.extend(list(incompatible.missing_keys))
        unexpected_keys.extend(list(incompatible.unexpected_keys))
    elif "dit" in payload and hasattr(model, "video_expert"):
        logging.warning("Loading legacy `dit` checkpoint into video expert only.")
        incompatible = model.video_expert.load_state_dict(payload["dit"], strict=False)
        missing_keys.extend(list(incompatible.missing_keys))
        unexpected_keys.extend(list(incompatible.unexpected_keys))
    else:
        raise ValueError(f"Checkpoint missing supported model weights (`mot` or `dit`): {ckpt}")

    if getattr(model, "proprio_encoder", None) is not None:
        if "proprio_encoder" in payload:
            incompatible = model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            missing_keys.extend(list(incompatible.missing_keys))
            unexpected_keys.extend(list(incompatible.unexpected_keys))
        else:
            logging.warning(
                "Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params."
            )
    elif "proprio_encoder" in payload:
        logging.warning(
            "Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring."
        )

    logging.info(
        "Loaded checkpoint: %s | payload_keys=%s | step=%s | missing_keys=%s | unexpected_keys=%s",
        ckpt,
        sorted(payload.keys()),
        payload.get("step"),
        missing_keys,
        unexpected_keys,
    )


def _resolve_default_navsim_root() -> Path:
    return project_root / "data" / "navsim"


_NAVSIM_DEVKIT_ROOT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", project_root / "navsim"))
_SCENE_FILTER_DIR = (
    _NAVSIM_DEVKIT_ROOT
    / "navsim"
    / "planning"
    / "script"
    / "config"
    / "common"
    / "train_test_split"
    / "scene_filter"
)


def _resolve_scene_filter_path(value: str) -> str:
    """Resolve a scene_filter reference (a yaml path OR a bare filter name).

    - If ``value`` points to an existing file, use it as-is.
    - Otherwise treat it as a filter name under the standard scene_filter dir,
      e.g. ``navtrain_hardsub`` -> ``<scene_filter_dir>/navtrain_hardsub.yaml``.
    """
    resolved = os.path.expanduser(os.path.expandvars(str(value)))
    candidate = Path(resolved)
    if candidate.exists():
        return str(candidate)
    name = resolved[:-5] if resolved.endswith(".yaml") else resolved
    return str(_SCENE_FILTER_DIR / f"{name}.yaml")


def _apply_navsim_env_overrides(eval_cfg: DictConfig) -> None:
    """Let environment variables override split / scene_filter without touching launch scripts.

    - ``NAVSIM_SPLIT``: overrides ``EVALUATION.split`` (e.g. ``navtrain`` / ``navtest``).
    - ``NAVSIM_SCENE_FILTER``: overrides ``EVALUATION.scene_filter`` to run only a subset
      of scenes (path or bare filter name). Unset -> full split (current behavior).
    """
    split_env = os.environ.get("NAVSIM_SPLIT")
    if split_env:
        eval_cfg.split = str(split_env).strip()
    scene_filter_env = os.environ.get("NAVSIM_SCENE_FILTER")
    if scene_filter_env:
        eval_cfg.scene_filter = _resolve_scene_filter_path(scene_filter_env)


def _resolve_split_defaults(split_name: str) -> tuple[str, Path, str]:
    split_key = str(split_name).strip().lower()
    if split_key == "navtest":
        scene_filter = _SCENE_FILTER_DIR / "navtest.yaml"
        return "test_logs", scene_filter, "test"
    if split_key in {"navsim", "navval", "val"}:
        scene_filter = _SCENE_FILTER_DIR / "navtrain.yaml"
        return "val_logs", scene_filter, "trainval"
    if split_key in {"navtrain", "train"}:
        scene_filter = _SCENE_FILTER_DIR / "navtrain.yaml"
        return "train_logs", scene_filter, "trainval"
    raise ValueError(f"Unsupported EVALUATION.split={split_name}. Expected one of ['navsim', 'navtrain', 'navtest'].")


def _ensure_eval_defaults(cfg: DictConfig) -> None:
    if "EVALUATION" not in cfg or cfg.EVALUATION is None:
        cfg.EVALUATION = OmegaConf.create({})
    eval_cfg = cfg.EVALUATION
    defaults = {
        "split": "navsim",
        "output_dir": None,
        "max_samples": None,
        "save_videos": True,
        "save_video_limit": 50,
        "save_pred_only_video": False,
        "infer_mode": "auto",
        "num_inference_steps": None,
        "sigma_shift": None,
        "text_cfg_scale": 1.0,
        "negative_prompt": "",
        "rand_device": "cpu",
        "seed": 0,
        "tiled": False,
        "device": None,
        "navsim_root": None,
        "navsim_log_path": None,
        "sensor_blobs_path": None,
        "scene_filter": None,
        "split_logs_key": None,
        "save_actions": True,
        "action_output_dir": None,
    }
    for key, value in defaults.items():
        if key not in eval_cfg or eval_cfg.get(key) is None:
            eval_cfg[key] = value

    # Environment overrides win over config defaults (used to select a subset of scenes
    # or a different split without editing the launch scripts).
    _apply_navsim_env_overrides(eval_cfg)

    if eval_cfg.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        eval_cfg.output_dir = str(project_root / "evaluate_results" / "navsim" / str(hydra.core.hydra_config.HydraConfig.get().job.config_name) / timestamp)

def _resolve_navsim_paths(cfg: DictConfig) -> tuple[Path, Path, Path, str]:
    eval_cfg = cfg.EVALUATION
    split_logs_key, default_scene_filter, data_subdir = _resolve_split_defaults(str(eval_cfg.split))
    navsim_root = (
        Path(os.path.expanduser(os.path.expandvars(str(eval_cfg.navsim_root))))
        if eval_cfg.navsim_root is not None
        else _resolve_default_navsim_root()
    )
    navsim_log_path = (
        Path(os.path.expanduser(os.path.expandvars(str(eval_cfg.navsim_log_path))))
        if eval_cfg.navsim_log_path is not None
        else navsim_root / "navsim_logs" / data_subdir
    )
    sensor_blobs_path = (
        Path(os.path.expanduser(os.path.expandvars(str(eval_cfg.sensor_blobs_path))))
        if eval_cfg.sensor_blobs_path is not None
        else navsim_root / "sensor_blobs" / data_subdir
    )
    scene_filter = (
        Path(os.path.expanduser(os.path.expandvars(str(eval_cfg.scene_filter))))
        if eval_cfg.scene_filter is not None
        else default_scene_filter
    )
    split_logs_key = str(eval_cfg.split_logs_key) if eval_cfg.split_logs_key is not None else split_logs_key
    return navsim_log_path, sensor_blobs_path, scene_filter, split_logs_key


def _build_eval_dataset(cfg: DictConfig):
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    navsim_log_path, sensor_blobs_path, scene_filter, split_logs_key = _resolve_navsim_paths(cfg)
    dataset_cfg.navsim_log_path = str(navsim_log_path)
    dataset_cfg.sensor_blobs_path = str(sensor_blobs_path)
    dataset_cfg.scene_filter = str(scene_filter)
    dataset_cfg.split_logs_key = split_logs_key
    dataset_cfg.is_training_set = False
    return instantiate(dataset_cfg)


def _to_uint8_image(image_chw: torch.Tensor) -> np.ndarray:
    x = image_chw.detach().float().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8)
    return x.permute(1, 2, 0).contiguous().numpy()


def _save_side_by_side_video(
    pred_frames: list[Image.Image],
    gt_video: torch.Tensor,
    output_path: Path,
) -> None:
    gt_frames = [_to_uint8_image(gt_video[:, t]) for t in range(gt_video.shape[1])]
    stitched_frames: list[Image.Image] = []
    for pred_frame, gt_frame in zip(pred_frames, gt_frames):
        pred_rgb = np.array(pred_frame.convert("RGB"), dtype=np.uint8)
        stitched = np.concatenate([pred_rgb, gt_frame], axis=1)
        stitched_frames.append(Image.fromarray(stitched))
    save_mp4(stitched_frames, str(output_path), fps=2)


def _save_pred_video(pred_frames: list[Image.Image], output_path: Path) -> None:
    save_mp4(list(pred_frames), str(output_path), fps=2)


def _save_gt_video(gt_video: torch.Tensor, output_path: Path) -> None:
    """Save the ground-truth clip as a standalone mp4 (single panel, same codec as
    the prediction) so FVD/FID can use it as the real reference set. ``gt_video`` is
    [C, T, H, W] in [-1, 1]."""
    gt_frames = [Image.fromarray(_to_uint8_image(gt_video[:, t])) for t in range(gt_video.shape[1])]
    save_mp4(gt_frames, str(output_path), fps=2)


def _build_infer_kwargs(
    *,
    model: torch.nn.Module,
    sample: dict[str, Any],
    cfg: DictConfig,
    action_horizon: int,
    num_video_frames: int,
) -> dict[str, Any]:
    eval_cfg = cfg.EVALUATION
    num_inference_steps_cfg = eval_cfg.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)

    input_image = sample["video"][:, 0].unsqueeze(0)
    proprio = sample.get("proprio")
    if proprio is not None:
        if proprio.ndim != 2:
            raise ValueError(f"Expected sample['proprio'] to be [T, D], got {tuple(proprio.shape)}")
        proprio = proprio[0]

    infer_kwargs = {
        "prompt": None,
        "context": sample.get("context"),
        "context_mask": sample.get("context_mask"),
        "input_image": input_image,
        "action_horizon": int(action_horizon),
        "negative_prompt": str(eval_cfg.get("negative_prompt", "")),
        "text_cfg_scale": float(eval_cfg.get("text_cfg_scale", 1.0)),
        "num_inference_steps": int(num_inference_steps),
        "sigma_shift": None if eval_cfg.get("sigma_shift") is None else float(eval_cfg.get("sigma_shift")),
        "seed": None if eval_cfg.get("seed") is None else int(eval_cfg.get("seed")),
        "rand_device": str(eval_cfg.get("rand_device", "cpu")),
        "tiled": bool(eval_cfg.get("tiled", False)),
        "proprio": proprio,
    }
    if infer_kwargs["context"] is None or infer_kwargs["context_mask"] is None:
        infer_kwargs["prompt"] = sample["prompt"]
        infer_kwargs.pop("context", None)
        infer_kwargs.pop("context_mask", None)

    infer_mode = str(eval_cfg.get("infer_mode", "auto")).strip().lower()
    if infer_mode == "auto":
        infer_mode = "action_only" if str(eval_cfg.split).strip().lower() == "navtest" else "joint"
    if infer_mode not in {"joint", "action_only"}:
        raise ValueError(f"Unsupported infer_mode={infer_mode}. Expected one of ['auto', 'joint', 'action_only'].")

    if infer_mode == "joint":
        infer_kwargs["num_video_frames"] = int(num_video_frames)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = int(num_video_frames)

    return {"mode": infer_mode, "kwargs": infer_kwargs}


@torch.no_grad()
def _run_single_inference(
    *,
    model: torch.nn.Module,
    sample: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    num_video_frames = int(sample["video"].shape[1])
    action_horizon = int(sample["action"].shape[0])
    plan = _build_infer_kwargs(
        model=model,
        sample=sample,
        cfg=cfg,
        action_horizon=action_horizon,
        num_video_frames=num_video_frames,
    )
    infer_mode = plan["mode"]
    infer_kwargs = plan["kwargs"]

    if infer_mode == "joint":
        return model.infer_joint(**infer_kwargs)
    return model.infer_action(**infer_kwargs)


def _compute_action_metrics(
    *,
    dataset,
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
) -> dict[str, float]:
    pred_action = pred_action.detach().cpu().float()
    gt_action = gt_action.detach().cpu().float()
    if pred_action.ndim == 2:
        pred_action = pred_action.unsqueeze(0)
    if gt_action.ndim == 2:
        gt_action = gt_action.unsqueeze(0)

    pred_denorm = dataset.denormalize_action(pred_action)
    gt_denorm = dataset.denormalize_action(gt_action)
    diff = pred_denorm - gt_denorm

    xy_diff = diff[..., :2]
    heading_diff = diff[..., 2]
    ade = torch.linalg.norm(xy_diff, dim=-1).mean().item()
    fde = torch.linalg.norm(xy_diff[:, -1], dim=-1).mean().item()
    metrics = {
        "action_l1": diff.abs().mean().item(),
        "action_l2": diff.pow(2).mean().item(),
        "traj_ade": ade,
        "traj_fde": fde,
        "heading_mae": heading_diff.abs().mean().item(),
    }
    return metrics


def _compute_video_metrics(
    *,
    pred_frames: list[Image.Image],
    gt_video: torch.Tensor,
) -> dict[str, float]:
    pred_video_tensor = pil_frames_to_video_tensor(pred_frames)
    gt_video_tensor = ((gt_video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
    if pred_video_tensor.shape != gt_video_tensor.shape:
        raise ValueError(
            "Prediction/GT video shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )
    return {
        "video_psnr": float(video_psnr(pred=pred_video_tensor, target=gt_video_tensor)),
        "video_ssim": float(video_ssim(pred=pred_video_tensor, target=gt_video_tensor)),
    }

def _mean_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def _build_summary(
    *,
    eval_cfg: DictConfig,
    num_samples: int,
    per_sample_results: list[dict[str, Any]],
    per_sample_path: Path,
    actions_dir: Optional[Path],
) -> dict[str, Any]:
    action_l1_values = [float(row["action_l1"]) for row in per_sample_results if "action_l1" in row]
    action_l2_values = [float(row["action_l2"]) for row in per_sample_results if "action_l2" in row]
    traj_ade_values = [float(row["traj_ade"]) for row in per_sample_results if "traj_ade" in row]
    traj_fde_values = [float(row["traj_fde"]) for row in per_sample_results if "traj_fde" in row]
    heading_mae_values = [float(row["heading_mae"]) for row in per_sample_results if "heading_mae" in row]
    video_psnr_values = [float(row["video_psnr"]) for row in per_sample_results if "video_psnr" in row]
    video_ssim_values = [float(row["video_ssim"]) for row in per_sample_results if "video_ssim" in row]
    return {
        "split": str(eval_cfg.split),
        "num_samples": int(num_samples),
        "action_l1_mean": _mean_or_none(action_l1_values),
        "action_l2_mean": _mean_or_none(action_l2_values),
        "traj_ade_mean": _mean_or_none(traj_ade_values),
        "traj_fde_mean": _mean_or_none(traj_fde_values),
        "heading_mae_mean": _mean_or_none(heading_mae_values),
        "video_psnr_mean": _mean_or_none(video_psnr_values),
        "video_ssim_mean": _mean_or_none(video_ssim_values),
        "per_sample_results_path": str(per_sample_path),
        "pred_actions_dir": str(actions_dir) if actions_dir is not None else None,
    }


def evaluate_dataset(
    *,
    model: torch.nn.Module,
    dataset,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    eval_cfg = cfg.EVALUATION
    videos_dir = output_dir / "videos"
    if bool(eval_cfg.get("save_videos", True)):
        videos_dir.mkdir(parents=True, exist_ok=True)
    save_gt_video = bool(eval_cfg.get("save_gt_video", False))
    videos_gt_dir = output_dir / "videos_gt"
    if save_gt_video:
        videos_gt_dir.mkdir(parents=True, exist_ok=True)
    actions_dir = (
        Path(os.path.expanduser(os.path.expandvars(str(eval_cfg.action_output_dir))))
        if eval_cfg.get("action_output_dir") is not None
        else output_dir / "pred_actions"
    )
    if bool(eval_cfg.get("save_actions", True)):
        actions_dir.mkdir(parents=True, exist_ok=True)

    rank = _get_rank()
    world_size = _get_world_size()
    max_samples_cfg = eval_cfg.get("max_samples", None)
    num_samples = len(dataset) if max_samples_cfg is None else min(len(dataset), int(max_samples_cfg))
    # save_video_limit caps how many mp4s are written per rank; None or <= 0 means
    # save every generated video (PSNR/SSIM are computed for all samples regardless).
    _svl = eval_cfg.get("save_video_limit", 50)
    save_video_limit = None if _svl is None else int(_svl)
    save_all_videos = save_video_limit is None or save_video_limit <= 0
    save_pred_only_video = bool(eval_cfg.get("save_pred_only_video", False))
    per_sample_results: list[dict[str, Any]] = []
    sample_indices = list(range(rank, num_samples, world_size))
    progress = tqdm(
        sample_indices,
        desc=f"Eval {eval_cfg.split} rank{rank}",
        disable=not _is_main_process(),
    )
    shard_path = output_dir / f"per_sample_results.rank{rank:02d}.jsonl"

    for _loop_i, idx in enumerate(progress):
        sample = dataset[idx]
        token = str(sample.get("token", idx))
        pred = _run_single_inference(model=model, sample=sample, cfg=cfg)

        pred_action = pred["action"]
        gt_action = sample["action"]
        pred_action_bt3 = pred_action.unsqueeze(0) if pred_action.ndim == 2 else pred_action
        pred_action_denorm = dataset.denormalize_action(pred_action_bt3)
        sample_metrics = {
            "index": int(idx),
            "token": token,
        }
        sample_metrics.update(
            _compute_action_metrics(
                dataset=dataset,
                pred_action=pred_action,
                gt_action=gt_action,
            )
        )

        if bool(eval_cfg.get("save_actions", True)):
            action_path = actions_dir / f"{token}.npy"
            np.save(action_path, pred_action_denorm[0].numpy().astype(np.float32))
            sample_metrics["action_path"] = str(action_path)

        pred_video = pred.get("video", None)
        if pred_video is not None:
            video_metrics = _compute_video_metrics(pred_frames=pred_video, gt_video=sample["video"])
            sample_metrics.update(video_metrics)
            if bool(eval_cfg.get("save_videos", True)) and (save_all_videos or idx < save_video_limit):
                video_path = videos_dir / f"{token}.mp4"
                if save_pred_only_video:
                    _save_pred_video(pred_video, video_path)
                else:
                    _save_side_by_side_video(pred_video, sample["video"], video_path)
                sample_metrics["video_path"] = str(video_path)
            if save_gt_video and (save_all_videos or idx < save_video_limit):
                gt_video_path = videos_gt_dir / f"{token}.mp4"
                _save_gt_video(sample["video"], gt_video_path)
                sample_metrics["gt_video_path"] = str(gt_video_path)

        per_sample_results.append(sample_metrics)

        # Free per-sample objects and periodically force GC. NavSim scene objects transitively
        # hold OS file handles (sensor images, map/GPKG layers) released only when collected;
        # over a long eval, cyclic-GC delay lets them accumulate until the process hits its
        # open-file limit (OSError: [Errno 24] Too many open files).
        del sample, pred
        if (_loop_i + 1) % 20 == 0:
            gc.collect()

    with shard_path.open("w", encoding="utf-8") as f:
        for row in per_sample_results:
            f.write(json.dumps(row, cls=NumpyEncoder, ensure_ascii=True) + "\n")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not _is_main_process():
        return {
            "split": str(eval_cfg.split),
            "rank": rank,
            "world_size": world_size,
            "num_local_samples": len(per_sample_results),
            "per_sample_results_path": str(shard_path),
        }

    merged_results: list[dict[str, Any]] = []
    for shard_rank in range(world_size):
        cur_shard_path = output_dir / f"per_sample_results.rank{shard_rank:02d}.jsonl"
        if not cur_shard_path.exists():
            continue
        with cur_shard_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    merged_results.append(json.loads(line))
    merged_results.sort(key=lambda row: int(row["index"]))

    per_sample_path = output_dir / "per_sample_results.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in merged_results:
            f.write(json.dumps(row, cls=NumpyEncoder, ensure_ascii=True) + "\n")

    return _build_summary(
        eval_cfg=eval_cfg,
        num_samples=num_samples,
        per_sample_results=merged_results,
        per_sample_path=per_sample_path,
        actions_dir=actions_dir if bool(eval_cfg.get("save_actions", True)) else None,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_navsim.yaml",)
def main(cfg: DictConfig):
    _ensure_eval_defaults(cfg)
    dist_enabled = _init_distributed()
    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("ckpt") is None:
        raise ValueError("ckpt must not be None. Pass ckpt=/path/to/checkpoint.pt")

    rank = _get_rank()
    log_path = output_dir / ("eval.log" if _is_main_process() else f"eval.rank{rank:02d}.log")
    logging.basicConfig(
        level=logging.INFO if _is_main_process() else logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )

    start_time = time.time()
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset = _build_eval_dataset(cfg)
    if _is_main_process():
        OmegaConf.save(config=cfg, f=str(output_dir / "eval_config.yaml"))

    logging.info("Output directory: %s", output_dir)
    logging.info("Checkpoint: %s", cfg.ckpt)
    logging.info("Model device=%s dtype=%s rank=%d world_size=%d", model_device, model_dtype, rank, _get_world_size())
    logging.info("Dataset split=%s samples=%d", cfg.EVALUATION.split, len(dataset))

    summary = evaluate_dataset(
        model=model,
        dataset=dataset,
        cfg=cfg,
        output_dir=output_dir,
    )
    if _is_main_process():
        summary["duration_sec"] = float(time.time() - start_time)
        summary["output_dir"] = str(output_dir)
        summary["checkpoint"] = str(cfg.ckpt)
        summary["world_size"] = _get_world_size()

        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder, ensure_ascii=True)

        logging.info("Saved summary to %s", summary_path)
        logging.info("Summary: %s", json.dumps(summary, cls=NumpyEncoder, ensure_ascii=True))

    if dist_enabled and dist.is_initialized():
        dist.barrier()
    _cleanup_distributed()
    return summary


if __name__ == "__main__":
    _prefer_navsim_devkit()
    main()
