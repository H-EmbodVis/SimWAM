"""Native 20-point action inference for test frames without ground truth."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_waymo_test(model, dataset, accelerator, output_dir, num_inference_steps=10, seed=42, max_samples=None):
    if dataset.evaluation_mode != "prediction_only":
        raise ValueError("Use the label-free WaymoTestDataset")
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    count = len(dataset)
    if max_samples is not None:
        if int(max_samples) < 1:
            raise ValueError("max_samples must be positive")
        count = min(count, int(max_samples))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank, world = accelerator.process_index, accelerator.num_processes
    indices = range(rank, count, world)
    logger.info("Test rank=%d/%d: %d of %d frames", rank, world, len(indices), count)
    rank_path = output_dir / f"predictions_rank_{rank:03d}.jsonl"
    completed = 0
    with rank_path.open("w", encoding="utf-8") as handle:
        for index in indices:
            sample = dataset[index]
            if "action" in sample or "action_raw" in sample:
                raise ValueError("Prediction-only samples must not fabricate GT actions")
            video, proprio = sample["video"], sample["proprio"]
            if tuple(video.shape[:2]) != (3, 1) or tuple(proprio.shape) != (20, 8):
                raise ValueError("Test input must be [3,1,H,W] plus the current 8D state")
            with accelerator.autocast():
                normalized = model.infer_action(
                    prompt=None, input_image=video[:, 0], action_horizon=20,
                    proprio=proprio[0], context=sample["context"], context_mask=sample["context_mask"],
                    num_inference_steps=int(num_inference_steps), seed=int(seed)+index, tiled=False,
                )["action"]
            prediction = dataset.denormalize_action(torch.as_tensor(normalized))
            if prediction.shape == (1, 20, 2):
                prediction = prediction[0]
            if prediction.shape != (20, 2) or not torch.isfinite(prediction).all():
                raise ValueError(f"Invalid native XY prediction for {sample['token']}")
            record = {
                "dataset_index": index, "token": sample["token"], "frame_name": sample["frame_name"],
                "clip_id": sample["clip_id"], "current_frame_id": sample["current_frame_id"],
                "front_image": sample["front_image"],
                "predicted_traj": prediction.tolist(), "coordinate_frame": "ego_at_anchor_x_forward_y_left",
                "trajectory_hz": 4, "trajectory_horizon_s": 5, "action_dim": 2,
                "prediction_times_4hz": [i/4 for i in range(1, 21)],
                "seed": int(seed)+index,
            }
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            completed += 1
            if completed % 20 == 0 or completed == len(indices):
                handle.flush()
                logger.info("Test rank=%d completed %d/%d", rank, completed, len(indices))
    total = accelerator.reduce(torch.tensor(completed, dtype=torch.int64, device=accelerator.device), reduction="sum")
    if total.item() != count:
        raise RuntimeError(f"Expected {count} test predictions, got {total.item()}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged = []
        for other_rank in range(world):
            path = output_dir / f"predictions_rank_{other_rank:03d}.jsonl"
            merged.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        merged.sort(key=lambda record: record["dataset_index"])
        if [row["dataset_index"] for row in merged] != list(range(count)):
            raise RuntimeError("Duplicate or missing prediction indices")
        if [row["token"] for row in merged] != list(dataset._tokens[:count]):
            raise RuntimeError("Test prediction frame names do not match input order")
        if len({row["frame_name"] for row in merged}) != count:
            raise RuntimeError("Duplicate official test frame names")
        path = output_dir / "predictions.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        summary = {
            "mode": "action_only_test_prediction", "num_predictions": count,
            "num_input_frames": len(dataset), "complete_input_coverage": count == len(dataset),
            "world_size": world, "num_inference_steps": int(num_inference_steps), "base_seed": int(seed),
            "prediction_shape": [count, 20, 2], "prediction_grid_seconds": [i/4 for i in range(1, 21)],
            "prediction_units": "meters", "has_ground_truth": False,
            "test_metrics_available_locally": False,
            "dataset_jsonl": str(dataset.dataset_jsonl),
            "dataset_sha256": hashlib.sha256(Path(dataset.dataset_jsonl).read_bytes()).hexdigest(),
            "normalization_stats": str(dataset.norm_stats_path),
            "normalization_sha256": hashlib.sha256(Path(dataset.norm_stats_path).read_bytes()).hexdigest(),
            "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        (output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    accelerator.wait_for_everyone()
    return {"num_predictions": count, "complete_input_coverage": count == len(dataset)}
