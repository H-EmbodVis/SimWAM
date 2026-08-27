#!/usr/bin/env python3
import inspect
import hashlib
import json
import logging
import math
import os
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import av
import hydra
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as transforms_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
navsim_exp_dir = project_root / "experiments" / "navsim"
for p in (str(project_root), str(navsim_exp_dir), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_navsim import (
    NumpyEncoder,
    _get_rank,
    _get_world_size,
    _init_distributed,
    _is_main_process,
    _load_model_checkpoint,
    _mean_or_none,
    _mixed_precision_to_model_dtype,
    _resolve_eval_device,
)

logger = logging.getLogger(__name__)
CAM = "camera_front_wide_120fov"
WINDOW_STEPS_10HZ = 40
SCORE_HORIZONS_S = (1.0, 2.0, 3.0, 4.0)
_BOK_SUMMARY_KEYS = tuple(
    f"{agg}_{metric}_{mode}{suffix}"
    for suffix in ("_1s", "_2s", "_3s", "")
    for mode in ("10hz", "2hz")
    for metric in ("ade", "fde")
    for agg in ("min", "mean")
)
_LEGACY_SUMMARY_ALIASES = {
    "ade_10hz_mean": "mean_ade_10hz",
    "fde_10hz_mean": "mean_fde_10hz",
    "ade_2hz_mean": "mean_ade_2hz",
    "fde_2hz_mean": "mean_fde_2hz",
}


def _load_fixed_prompt_context(cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    from simwam.datasets.navsim.navsim_dataset import NavSimVideoDataset

    data_cfg = cfg.data.train
    if bool(data_cfg.get("use_dynamic_prompt", False)):
        raise ValueError("This evaluation requires use_dynamic_prompt=false.")
    context_len = int(data_cfg.get("context_len", 256))
    cache_dir = data_cfg.get("text_embedding_cache_dir") or os.environ.get(
        "NAVSIM_TEXT_EMBED_CACHE", "./data/text_embeds_cache/navsim"
    )
    prompt = NavSimVideoDataset.build_prompt_fixed(
        hist_xyh=torch.zeros(1, 3), high_cmd_one_hot=torch.zeros(4),
        speed_mps=0.0, acc_mps2=0.0, use_dynamic_prompt=False,
    )
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = Path(cache_dir) / f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing fixed-prompt T5 cache: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"].to(dtype=torch.bfloat16)
    context_mask = payload["mask"].bool()
    context[~context_mask] = 0
    return context.contiguous(), torch.ones_like(context_mask).contiguous()


def _load_physicalai_denorm(stats_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    traj_stats = stats.get("future_traj_10hz", stats)
    bias, rng = [], []
    for channel in ("x", "y", "heading"):
        q1 = float(traj_stats[channel]["q1"])
        q99 = float(traj_stats[channel]["q99"])
        if q99 <= q1:
            raise ValueError(f"Invalid q1/q99 for {channel}: {q1} / {q99}")
        bias.append(-q1)
        rng.append(q99 - q1)
    return torch.tensor(bias, dtype=torch.float32), torch.tensor(rng, dtype=torch.float32)


def denorm_physicalai(normalized: torch.Tensor, bias: torch.Tensor, rng: torch.Tensor) -> torch.Tensor:
    return (normalized + 1) / 2 * rng - bias


def _preprocess_image(image: Image.Image, video_size_hw: list[int]) -> torch.Tensor:
    tensor = transforms_F.to_tensor(image)
    tensor = transforms_F.resize(
        tensor, size=video_size_hw, interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
    )
    return transforms_F.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous()


def _score_best_of_k(pred_k: np.ndarray, gt10: np.ndarray, gt_fps: float, pred_fps: float) -> tuple[dict[str, Any], int]:
    _, n_pred, _ = pred_k.shape
    n_gt = gt10.shape[0]
    scores: dict[str, Any] = {}
    ade10_full = np.empty(pred_k.shape[0])
    for h_s in SCORE_HORIZONS_S:
        ngt = min(int(round(h_s * gt_fps)), n_gt)
        npred = min(int(round(h_s * pred_fps)), n_pred)
        if ngt <= 0 or npred <= 0:
            continue
        gt_times = np.arange(1, ngt + 1) / gt_fps
        ade10 = np.empty(pred_k.shape[0])
        fde10 = np.empty(pred_k.shape[0])
        ade2 = np.empty(pred_k.shape[0])
        fde2 = np.empty(pred_k.shape[0])
        for k in range(pred_k.shape[0]):
            p = pred_k[k, :npred]
            pline = np.concatenate([np.zeros((1, 2)), p], axis=0)
            pline_t = np.concatenate([[0.0], np.arange(1, npred + 1) / pred_fps])
            p10 = np.stack([np.interp(gt_times, pline_t, pline[:, c]) for c in range(2)], axis=-1)
            d10 = np.linalg.norm(p10 - gt10[:ngt], axis=-1)
            ade10[k], fde10[k] = d10.mean(), d10[-1]
            idx_pred2 = [min(int(round(i / 2.0 * pred_fps)) - 1, npred - 1) for i in range(1, int(2 * h_s) + 1)]
            idx_gt2 = [min(int(round(i / 2.0 * gt_fps)) - 1, ngt - 1) for i in range(1, int(2 * h_s) + 1)]
            d2 = np.linalg.norm(p[idx_pred2] - gt10[idx_gt2], axis=-1)
            ade2[k], fde2[k] = d2.mean(), d2[-1]
        suffix = "" if h_s == SCORE_HORIZONS_S[-1] else f"_{int(h_s)}s"
        scores.update({
            f"min_ade_10hz{suffix}": float(ade10.min()), f"mean_ade_10hz{suffix}": float(ade10.mean()),
            f"min_fde_10hz{suffix}": float(fde10.min()), f"mean_fde_10hz{suffix}": float(fde10.mean()),
            f"min_ade_2hz{suffix}": float(ade2.min()), f"mean_ade_2hz{suffix}": float(ade2.mean()),
            f"min_fde_2hz{suffix}": float(fde2.min()), f"mean_fde_2hz{suffix}": float(fde2.mean()),
        })
        if suffix == "":
            ade10_full = ade10
    return scores, int(np.argmin(ade10_full))


def _se3_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def _rel_xytheta(T_anchor_inv: np.ndarray, pose_k: np.ndarray) -> tuple[float, float, float]:
    rel = T_anchor_inv @ pose_k
    return float(rel[0, 3]), float(rel[1, 3]), float(np.arctan2(rel[1, 0], rel[0, 0]))


def list_clips(data_root: str) -> list[str]:
    root = Path(data_root)
    return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / f"{CAM}_ego.pkl").exists())


@lru_cache(maxsize=8)
def load_ego(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def num_windows(ego: dict) -> int:
    return max(0, (len(ego["timestamps_us"]) - 1) // WINDOW_STEPS_10HZ)


def decode_frames_at_indices(mp4_path: str, indices) -> dict[int, np.ndarray]:
    want = {int(i) for i in indices}
    if not want:
        return {}
    output = {}
    container = av.open(mp4_path)
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i in want:
                output[i] = frame.to_ndarray(format="rgb24")
            if i >= max(want):
                break
    finally:
        container.close()
    return output


def build_window_gt(ego: dict, window_idx: int) -> dict:
    poses = np.asarray(ego["ego_pose__world_from_rig"])
    vel = np.asarray(ego["ego_velocity_rig"])
    acc = np.asarray(ego["ego_acceleration_rig"])
    start = window_idx * WINDOW_STEPS_10HZ
    T_inv = _se3_inv(poses[start])
    R_anchor = poses[start, :3, :3]
    gt10 = np.zeros((WINDOW_STEPS_10HZ, 2), np.float32)
    for k in range(1, WINDOW_STEPS_10HZ + 1):
        gt10[k - 1, :2] = _rel_xytheta(T_inv, poses[start + k])[:2]
    return {
        "start": int(start), "gt10": gt10,
        "vel": (R_anchor.T @ vel[start])[:2].astype(np.float32),
        "acc": (R_anchor.T @ acc[start])[:2].astype(np.float32),
    }


def get_calib(ego: dict) -> dict | None:
    ci, ext = ego.get("camera_intrinsics"), ego.get("extrinsics__rig_from_cam")
    if ci is None or ext is None:
        return None
    ci = ci.item() if hasattr(ci, "item") and not isinstance(ci, dict) else ci
    try:
        return {
            "fw_poly": [float(x) for x in ci["fw_poly"]], "cx": float(ci["cx"]), "cy": float(ci["cy"]),
            "width": int(ci["width"]), "height": int(ci["height"]),
            "rig_from_cam": np.asarray(ext, dtype=np.float64),
        }
    except Exception:
        return None


def project_ego_xy_ftheta(local_xy, fw_poly, cx, cy, rig_from_cam, image_w, image_h, ground_z=0.0, theta_max=1.5):
    local_xy = np.asarray(local_xy, dtype=np.float64)
    if local_xy.ndim != 2 or local_xy.shape[0] == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.bool_)
    n = local_xy.shape[0]
    pts_rig = np.concatenate([local_xy[:, :2], np.full((n, 1), ground_z), np.ones((n, 1))], axis=1)
    pts_cam = (np.linalg.inv(np.asarray(rig_from_cam, dtype=np.float64)) @ pts_rig.T).T[:, :3]
    xc, yc, zc = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    rho = np.sqrt(xc * xc + yc * yc)
    theta = np.arctan2(rho, zc)
    radius = np.polynomial.polynomial.polyval(theta, np.asarray(fw_poly, dtype=np.float64))
    ux = np.where(rho > 1e-9, xc / np.maximum(rho, 1e-9), 0.0)
    uy = np.where(rho > 1e-9, yc / np.maximum(rho, 1e-9), 0.0)
    px, py = cx + radius * ux, cy + radius * uy
    valid = ((zc > 0) & (theta < theta_max) & (px >= 0) & (px <= image_w - 1) &
             (py >= 0) & (py <= image_h - 1) & np.isfinite(px) & np.isfinite(py))
    return np.stack([px, py], axis=1).astype(np.float32), valid.astype(np.bool_)


def _draw_polyline(draw, points, valid, color, width, radius):
    for i in range(1, points.shape[0]):
        if valid[i - 1] and valid[i]:
            draw.line([tuple(points[i - 1]), tuple(points[i])], fill=color, width=max(1, int(width)))
    for i in range(points.shape[0]):
        if valid[i]:
            x, y = float(points[i, 0]), float(points[i, 1])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def save_projection_viz(*, image, fw_poly, cx, cy, rig_from_cam, gt_xy, out_path, token, label=None, pred_xy=None):
    if fw_poly is None or rig_from_cam is None:
        return False
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(2, int(round(out.width / 320)))
    anchor = np.zeros((1, 2), dtype=np.float32)

    def draw_traj(xy, color, line_width):
        line = np.concatenate([anchor, np.asarray(xy, dtype=np.float32)[:, :2]], axis=0)
        points, valid = project_ego_xy_ftheta(line, fw_poly, cx, cy, rig_from_cam, out.width, out.height)
        _draw_polyline(draw, points, valid, color, line_width, max(2, int(round(line_width * 1.1))))

    draw_traj(gt_xy, (44, 160, 44), width + 1)
    if pred_xy is not None:
        draw_traj(pred_xy, (214, 39, 40), max(1, width - 1))
    text = f"GT=green Pred=red {token[:16]}"
    if label:
        text += f" {label}"
    draw.text((10, 10), text, fill=(255, 255, 0))
    out.save(out_path)
    return True


def _command_from_gt_traj(gt_xy: np.ndarray, heading_th_rad: float) -> np.ndarray:
    dx = float(gt_xy[-1, 0] - gt_xy[-2, 0])
    dy = float(gt_xy[-1, 1] - gt_xy[-2, 1])
    heading = float(np.arctan2(dy, dx))
    if heading >= heading_th_rad:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if heading <= -heading_th_rad:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def _ensure_eval_defaults(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    if "EVALUATION" not in cfg or cfg.EVALUATION is None:
        cfg.EVALUATION = OmegaConf.create({})
    eval_cfg = cfg.EVALUATION
    defaults = {
        "data_root": os.environ.get("PHYSICALAI_DATA_ROOT", "./data/physicalai/front"),
        "window_steps_10hz": 40,
        "gt_fps": 10,
        "pred_fps": 10,
        "pred_horizon_s": 4.0,
        "flip_lateral": False,
        "cmd_heading_th": 0.31,
        "max_clips": None,
        "num_traj_samples": 1,
        "num_inference_steps": None,
        "sigma_shift": None,
        "seed": 0,
        "rand_device": "cpu",
        "tiled": False,
        "device": "cuda",
        "save_actions": True,
        "action_output_dir": None,
        "save_viz": True,
        "viz_dir": None,
        "viz_max_samples": 50,
        "output_dir": None,
    }
    for key, value in defaults.items():
        if key not in eval_cfg or eval_cfg.get(key) is None:
            eval_cfg[key] = value

    if eval_cfg.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        try:
            job_name = str(hydra.core.hydra_config.HydraConfig.get().runtime.choices["task"])
        except Exception:
            job_name = "physicalai"
        exp = str(cfg.get("experiment_name", None) or "physicalai_eval")
        eval_cfg.output_dir = str(Path("evaluate_results") / "physicalai" / job_name / exp / timestamp)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_physicalai_eval.yaml")
def main(cfg: DictConfig) -> None:
    _ensure_eval_defaults(cfg)
    _init_distributed()
    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)

    rank, world_size = _get_rank(), _get_world_size()
    logging.basicConfig(
        level=logging.INFO if _is_main_process() else logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / (f"eval.rank{rank:02d}.log" if not _is_main_process() else "eval.log"), encoding="utf-8"),
        ],
        force=True,
    )

    if cfg.get("ckpt") is None:
        raise ValueError("ckpt must not be None. Pass ckpt=./weights/SimWAM.pt")

    start = time.time()
    eval_cfg = cfg.EVALUATION
    flip = bool(eval_cfg.get("flip_lateral", False))
    gt_fps, pred_fps = float(eval_cfg.gt_fps), float(eval_cfg.pred_fps)

    device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(device).eval()
    if not hasattr(model, "infer_action"):
        raise ValueError(f"Model {type(model).__name__} has no `infer_action`.")
    mask_mode = str(getattr(model, "mot_attention_mask_mode", "isolated"))
    if mask_mode != "isolated":
        logger.warning("mot_attention_mask_mode=%s: action-only inference is train/test consistent only "
                       "for 'isolated'. Proceeding.", mask_mode)
    if str(cfg.data.train.get("trajectory_mode", "absolute")).strip().lower() != "absolute":
        raise ValueError("This script assumes trajectory_mode='absolute'.")

    context, context_mask = _load_fixed_prompt_context(cfg)
    norm_stats_path = str(cfg.data.train.get("norm_stats_path", "./data/physicalai_dataset_stats.json"))
    norm_bias, norm_range = _load_physicalai_denorm(norm_stats_path)
    video_size_hw = [int(v) for v in cfg.data.train.video_size]
    action_horizon = int(cfg.data.train.get("future_action_horizon", 40))
    expected_pts = int(round(float(eval_cfg.get("pred_horizon_s", 4.0)) * pred_fps))
    if expected_pts != action_horizon:
        raise ValueError(
            f"pred_fps={pred_fps} x pred_horizon_s={eval_cfg.get('pred_horizon_s', 4.0)} = {expected_pts} pts, "
            f"but the model predicts action_horizon={action_horizon} pts. Set EVALUATION.pred_fps accordingly."
        )
    cmd_heading_th = float(eval_cfg.get("cmd_heading_th", 0.31))

    data_root = str(eval_cfg.data_root)
    clips = list_clips(data_root)
    if eval_cfg.get("max_clips") is not None:
        clips = clips[: max(0, int(eval_cfg.max_clips))]
    if len(clips) == 0:
        raise RuntimeError(f"No clips with {CAM}_ego.pkl under {data_root}")

    actions_dir = None
    if bool(eval_cfg.get("save_actions", True)):
        actions_dir = (
            Path(os.path.expanduser(os.path.expandvars(str(eval_cfg.action_output_dir))))
            if eval_cfg.get("action_output_dir") is not None
            else output_dir / "pred_actions"
        )
        actions_dir.mkdir(parents=True, exist_ok=True)

    viz_root = None
    if bool(eval_cfg.get("save_viz", True)):
        viz_root = (
            Path(os.path.expanduser(os.path.expandvars(str(eval_cfg.viz_dir))))
            if eval_cfg.get("viz_dir") is not None else output_dir / "viz"
        )
        viz_root.mkdir(parents=True, exist_ok=True)

    my_clips = clips[rank::world_size]
    num_inference_steps = eval_cfg.get("num_inference_steps", None)
    if num_inference_steps is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    num_inference_steps = int(num_inference_steps)
    seed = eval_cfg.get("seed", None)
    num_traj_samples = max(1, int(eval_cfg.get("num_traj_samples", 1)))
    batch_best_of = "num_samples" in inspect.signature(model.infer_action).parameters

    if _is_main_process():
        logger.info("model=%s device=%s dtype=%s world_size=%d", type(model).__name__, device, model_dtype, world_size)
        logger.info("clips=%d (rank shard=%d) data_root=%s output_dir=%s", len(clips), len(my_clips), data_root, output_dir)
        logger.info("video_size=%s action_horizon=%d pred_fps=%g norm_stats=%s",
                    video_size_hw, action_horizon, pred_fps, norm_stats_path)
        logger.info("num_traj_samples=%d (best-of-N, batched=%s) seed=%s", num_traj_samples, batch_best_of, seed)

    per_window: list[dict[str, Any]] = []
    viz_max = int(eval_cfg.get("viz_max_samples", 50) or 0)
    viz_per_rank = -1 if (viz_root is None or viz_max <= 0) else math.ceil(viz_max / world_size)
    est_windows = max(1, len(my_clips) * 5)
    win_stride = 1 if viz_per_rank < 0 else max(1, est_windows // max(1, viz_per_rank))
    win_counter = 0
    n_viz = 0
    progress = tqdm(my_clips, desc=f"eval-pai rank{rank}", disable=not _is_main_process(), dynamic_ncols=True)
    for cid in progress:
        clip_dir = Path(data_root) / cid
        try:
            ego = load_ego(str(clip_dir / f"{CAM}_ego.pkl"))
            nw = num_windows(ego)
            starts = [w * int(eval_cfg.window_steps_10hz) for w in range(nw)]
            frames = decode_frames_at_indices(str(clip_dir / f"{CAM}.mp4"), starts)
        except Exception as exc:
            per_window.append({"clip_id": cid, "window": -1, "valid": False, "error": repr(exc)[:200]})
            logger.error("clip=%s LOAD FAIL: %s", cid[:8], exc)
            continue
        clip_calib = get_calib(ego) if viz_root is not None else None

        for w in range(nw):
            token = f"{cid}#w{w}"
            do_viz = viz_root is not None and (viz_per_rank < 0 or (n_viz < viz_per_rank and win_counter % win_stride == 0))
            win_counter += 1
            try:
                win = build_window_gt(ego, w)
                if win["start"] not in frames:
                    raise RuntimeError(f"missing decoded frame {win['start']}")
                image = Image.fromarray(frames[win["start"]])
                input_image = _preprocess_image(image, video_size_hw).unsqueeze(0)

                gt10 = win["gt10"].astype(np.float64)
                vel = win["vel"].astype(np.float64)
                acc = win["acc"].astype(np.float64)
                if flip:
                    gt10[:, 1] = -gt10[:, 1]
                    vel[1] = -vel[1]
                    acc[1] = -acc[1]
                command = _command_from_gt_traj(gt10, cmd_heading_th)
                proprio = torch.from_numpy(np.concatenate([vel, acc, command]).astype(np.float32))

                infer_kwargs = dict(
                    prompt=None, input_image=input_image, proprio=proprio,
                    context=context, context_mask=context_mask, action_horizon=action_horizon,
                    num_inference_steps=num_inference_steps,
                    sigma_shift=None if eval_cfg.get("sigma_shift") is None else float(eval_cfg.get("sigma_shift")),
                    rand_device=str(eval_cfg.get("rand_device", "cpu")),
                    tiled=bool(eval_cfg.get("tiled", False)),
                )
                base_seed = None if seed is None else int(seed)
                if batch_best_of:
                    actions = model.infer_action(seed=base_seed, num_samples=num_traj_samples, **infer_kwargs)["action"]
                else:
                    actions = torch.stack([
                        model.infer_action(seed=None if base_seed is None else base_seed + i, **infer_kwargs)["action"]
                        for i in range(num_traj_samples)
                    ], dim=0)
                actions = actions.detach().cpu().float()
                if actions.ndim == 2:
                    actions = actions.unsqueeze(0)
                pred_xyh = denorm_physicalai(actions, norm_bias, norm_range).numpy()
                if actions_dir is not None:
                    save_xyh = pred_xyh[0] if pred_xyh.shape[0] == 1 else pred_xyh
                    np.save(actions_dir / f"{token.replace('#', '_')}.npy", save_xyh.astype(np.float32))

                row: dict[str, Any] = {"clip_id": cid, "window": w, "token": token, "valid": True,
                                       "command": ["left", "straight", "right", "unknown"][int(command.argmax())]}
                pred_k = pred_xyh[:, :, :2].astype(np.float64)
                scores, best_idx = _score_best_of_k(pred_k, gt10, gt_fps, pred_fps)
                row.update({"num_samples": int(pred_k.shape[0]), "best_idx": best_idx})
                row.update(scores)
                if pred_k.shape[0] == 1:
                    row.update({"ade_10hz": scores["mean_ade_10hz"], "fde_10hz": scores["mean_fde_10hz"],
                                "ade_2hz": scores["mean_ade_2hz"], "fde_2hz": scores["mean_fde_2hz"]})

                if do_viz and clip_calib is not None:
                    best_xy = pred_k[best_idx]
                    idx_pred2 = [min(int(round(k / 2.0 * pred_fps)) - 1, best_xy.shape[0] - 1) for k in range(1, 9)]
                    idx_gt2 = [min(int(round(k / 2.0 * gt_fps)) - 1, gt10.shape[0] - 1) for k in range(1, 9)]
                    saved = save_projection_viz(
                        image=image, fw_poly=clip_calib["fw_poly"], cx=clip_calib["cx"], cy=clip_calib["cy"],
                        rig_from_cam=clip_calib["rig_from_cam"], pred_xy=best_xy[idx_pred2], gt_xy=gt10[idx_gt2],
                        out_path=viz_root / f"{token.replace('#', '_')}.jpg", token=token,
                        label=f"minADE10={scores['min_ade_10hz']:.2f}m",
                    )
                    if saved:
                        row["viz_path"] = str(viz_root / f"{token.replace('#', '_')}.jpg")
                        n_viz += 1
            except Exception as exc:
                row = {"clip_id": cid, "window": w, "token": token, "valid": False, "error": repr(exc)[:200]}
                logger.error("token=%s FAIL: %s", token, exc)
            per_window.append(row)

        ok = [r for r in per_window if r.get("valid")]
        if ok:
            progress.set_postfix(ade10=f"{np.mean([r['mean_ade_10hz'] for r in ok]):.3f}",
                                 minfde10=f"{np.mean([r['min_fde_10hz'] for r in ok]):.3f}",
                                 n=len(ok))
    progress.close()

    shard_path = output_dir / f"per_window.rank{rank:02d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as f:
        for r in per_window:
            f.write(json.dumps(r, cls=NumpyEncoder, ensure_ascii=True) + "\n")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if _is_main_process():
        merged: list[dict[str, Any]] = []
        for shard in sorted(output_dir.glob("per_window.rank*.jsonl")):
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        merged.append(json.loads(line))
        per_window_path = output_dir / "per_window.jsonl"
        with per_window_path.open("w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, cls=NumpyEncoder, ensure_ascii=True) + "\n")

        valid = [r for r in merged if r.get("valid")]

        def _col(name: str) -> list[float]:
            return [float(r[name]) for r in valid if name in r and r[name] is not None and np.isfinite(float(r[name]))]

        summary: dict[str, Any] = {
            "dataset": "physicalai",
            "infer_mode": "action_only",
            "num_windows": int(len(merged)),
            "num_valid": int(len(valid)),
            "num_failed": int(len(merged) - len(valid)),
            "num_clips": int(len({r["clip_id"] for r in merged})),
            "action_horizon": int(action_horizon),
            "pred_fps": float(pred_fps),
            "num_traj_samples": int(num_traj_samples),
            "batch_best_of": bool(batch_best_of),
        }
        for name in _BOK_SUMMARY_KEYS:
            summary[name] = _mean_or_none(_col(name))
        for legacy, new in _LEGACY_SUMMARY_ALIASES.items():
            summary[legacy] = summary[new]
        summary.update({
            "num_viz": int(sum(1 for r in valid if r.get("viz_path"))),
            "viz_dir": str(viz_root) if viz_root is not None else None,
            "duration_sec": float(time.time() - start),
            "output_dir": str(output_dir),
            "checkpoint": str(cfg.ckpt),
            "per_window_results_path": str(per_window_path),
        })
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder, ensure_ascii=True)
        for h_s in SCORE_HORIZONS_S:
            suffix = "" if h_s == SCORE_HORIZONS_S[-1] else f"_{int(h_s)}s"
            logger.info("[pai] @%.0fs 10Hz | ADE=%s FDE=%s | minADE=%s minFDE=%s",
                        h_s, summary[f"mean_ade_10hz{suffix}"], summary[f"mean_fde_10hz{suffix}"],
                        summary[f"min_ade_10hz{suffix}"], summary[f"min_fde_10hz{suffix}"])
        logger.info("[pai] @4s 2Hz  | ADE=%s FDE=%s | minADE=%s minFDE=%s",
                    summary["mean_ade_2hz"], summary["mean_fde_2hz"],
                    summary["min_ade_2hz"], summary["min_fde_2hz"])
        logger.info("Done in %.1fs. windows valid=%d/%d clips=%d failed=%d Summary: %s",
                    time.time() - start, len(valid), len(merged), summary["num_clips"], summary["num_failed"],
                    json.dumps(summary, cls=NumpyEncoder))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
