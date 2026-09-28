"""Shared current-frame evaluation used by training and experiments/waymo."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .metrics import SCORE_NAMES, rfs_provenance, score_waymo_predictions

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_waymo_actions(
    model,
    dataset,
    accelerator,
    output_dir: str | Path,
    num_inference_steps: int = 10,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict:
    """Evaluate each val row exactly once, without future video or training_loss."""
    if getattr(dataset, "evaluation_mode", None) != "action_only":
        raise ValueError("Waymo action evaluation requires the current-only validation dataset")
    if int(num_inference_steps) <= 0:
        raise ValueError("num_inference_steps must be positive")
    count = len(dataset)
    if max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError("max_samples must be positive")
        count = min(count, int(max_samples))
    if count == 0:
        raise ValueError("Validation dataset is empty")
    rank, world_size = int(accelerator.process_index), int(accelerator.num_processes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, predicted, targets, preferences, velocities = [], [], [], [], []
    indices = range(rank, count, world_size)
    logger.info("Waymo action val rank=%d/%d: %d local / %d total samples", rank, world_size, len(indices), count)
    for index in indices:
        sample = dataset[index]
        video = sample["video"]
        if video.ndim != 4 or tuple(video.shape[:2]) != (3, 1):
            raise ValueError("Action-only val must return video [3,1,H,W]")
        proprio = sample["proprio"]
        if tuple(proprio.shape) != (20, 8):
            raise ValueError("Expected the current 8D ego state repeated over 20 action points")
        with accelerator.autocast():
            prediction = model.infer_action(
                prompt=None,
                input_image=video[:, 0],
                action_horizon=20,
                proprio=proprio[0],
                context=sample["context"],
                context_mask=sample["context_mask"],
                num_inference_steps=int(num_inference_steps),
                seed=int(seed) + index,
                tiled=False,
            )["action"]
        prediction = dataset.denormalize_action(torch.as_tensor(prediction))
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if "action_raw" in sample:
            target = torch.as_tensor(sample["action_raw"]).detach().cpu().float()
        else:
            target = dataset.denormalize_action(sample["action"])
        if tuple(prediction.shape) != (20, 2) or tuple(target.shape) != (20, 2):
            raise ValueError("Predictions and GT must be 20 native XY points")
        reference = sample.get("preference_trajectories") or []
        velocity = proprio[0, :2].detach().cpu().float().numpy()
        predicted.append(prediction.numpy())
        targets.append(target.numpy())
        preferences.append(reference)
        velocities.append(velocity)
        records.append({
            "dataset_index": index,
            "token": sample["token"],
            "predicted_traj": prediction.tolist(),
            "future_traj_gt": target.tolist(),
            "preference_trajectories": reference,
            "ego_velocity": velocity.tolist(),
            "seed": int(seed) + index,
            "trajectory_hz": 4,
            "trajectory_horizon_s": 5,
            "action_dim": 2,
        })
        if len(records) % 20 == 0 or len(records) == len(indices):
            logger.info("Waymo action val rank=%d: completed %d/%d", rank, len(records), len(indices))

    if records:
        _, per_sample = score_waymo_predictions(
            np.stack(predicted), np.stack(targets), preferences, np.stack(velocities)
        )
        local_sums = [float(per_sample[name].sum()) for name in SCORE_NAMES]
        for i, record in enumerate(records):
            record["metrics"] = {name: float(per_sample[name][i]) for name in SCORE_NAMES}
    else:
        local_sums = [0.0] * len(SCORE_NAMES)

    rank_path = output_dir / f"predictions_rank_{rank:03d}.jsonl"
    with rank_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    totals = torch.tensor(local_sums + [len(records)], dtype=torch.float64, device=accelerator.device)
    totals = accelerator.reduce(totals, reduction="sum").cpu()
    evaluated = int(round(totals[-1].item()))
    if evaluated != count:
        raise RuntimeError(f"Expected {count} validation samples, evaluated {evaluated}")
    metrics = {name: float(totals[i].item() / evaluated) for i, name in enumerate(SCORE_NAMES)}
    metrics.update(eval_mode="action_only", num_samples=evaluated, num_pref_evaluated=evaluated)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged = []
        for other_rank in range(world_size):
            with (output_dir / f"predictions_rank_{other_rank:03d}.jsonl").open(encoding="utf-8") as handle:
                merged.extend(json.loads(line) for line in handle if line.strip())
        merged.sort(key=lambda record: record["dataset_index"])
        if [r["dataset_index"] for r in merged] != list(range(count)):
            raise RuntimeError("Prediction shards contain duplicate or missing validation indices")
        if len({r["token"] for r in merged}) != count:
            raise RuntimeError("Prediction shards contain duplicate validation tokens")
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for record in merged:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        report = {
            **metrics,
            "prediction_grid_seconds": [i / 4 for i in range(1, 21)],
            "num_inference_steps": int(num_inference_steps),
            "base_seed": int(seed),
            "normalization_stats": getattr(dataset, "norm_stats_path", None),
            "rfs": rfs_provenance(),
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    return metrics
