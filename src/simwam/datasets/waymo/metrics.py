"""Waymo native-4-Hz ADE/FDE and the unmodified official Rater Feedback Score."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ._vendor.rater_feedback_utils import get_rater_feedback_score

SCORE_NAMES = (
    "ADE", "FDE", "ADE@1s", "FDE@1s", "ADE@3s", "FDE@3s", "ADE@5s", "FDE@5s",
    "RFS", "RFS_within_trust_pct", "ADE_vs_best_pref", "ADE_vs_best_pref@3s",
    "action_l1", "action_l2",
)
RFS_SETTINGS = {
    "frequency": 4,
    "length_seconds": 5,
    "lat_lng_threshold_multipliers": (1.0, 4.0),
    "decay_factor": 0.1,
    "default_num_of_rater_specified_trajectories": 3,
    "minimum_score_outside_trust_region": 4.0,
}


def rfs_provenance() -> dict:
    path = Path(__file__).parent / "_vendor/rater_feedback_utils.py"
    return {"implementation": "official_waymo_rater_feedback_utils",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "settings": RFS_SETTINGS}


def pad_or_truncate_reference(xy: np.ndarray, count: int = 20) -> np.ndarray:
    """Match official/Qwen reference handling; never interpolate predictions."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Reference trajectory must contain finite XY points")
    if len(xy) >= count:
        return xy[:count].copy()
    return np.concatenate([xy, np.repeat(xy[-1:], count - len(xy), axis=0)], axis=0)


def score_waymo_predictions(predicted_xy, ground_truth_xy, preferences, initial_velocity):
    """Return aggregate metrics and per-sample arrays in physical units.

    Prediction/GT: [N,20,2], t=0.25,...,5 s; initial_velocity: [N,2] in m/s.
    Preferences: native lists of {pos_x,pos_y,preference_score}. They are passed
    unchanged to the official scorer, which truncates/pads references to 20.
    One model trajectory per sample, probability 1; no best-of-N label selection.
    """
    pred = np.asarray(predicted_xy, dtype=np.float64)
    gt = np.asarray(ground_truth_xy, dtype=np.float64)
    velocity = np.asarray(initial_velocity, dtype=np.float64)
    if pred.ndim != 3 or pred.shape[1:] != (20, 2) or not len(pred):
        raise ValueError("Predictions must be [N,20,2] on the native 4-Hz grid; no resampling is performed")
    if gt.shape != pred.shape or velocity.shape != (len(pred), 2):
        raise ValueError("GT/velocity shape does not match predictions")
    if not all(np.isfinite(x).all() for x in (pred, gt, velocity)):
        raise ValueError("Predictions, GT and velocity must be finite")
    if len(preferences) != len(pred):
        raise ValueError("Every prediction must have its reference trajectories")

    reference_batches, score_batches, best_references = [], [], []
    for items in preferences:
        if not items:
            raise ValueError("Missing scored reference trajectories; cannot compute RFS")
        refs, scores = [], []
        for item in items:
            x, y = item["pos_x"], item["pos_y"]
            if len(x) != len(y):
                raise ValueError("Reference x/y lengths differ")
            xy = np.column_stack([x, y]).astype(np.float64)
            if not len(xy) or not np.isfinite(xy).all():
                raise ValueError("Invalid reference XY")
            value = float(item["preference_score"])
            if not np.isfinite(value) or not 0 <= value <= 10:
                raise ValueError("Reference score must be in [0,10]")
            refs.append(xy)
            scores.append(value)
        reference_batches.append(refs)
        score_batches.append(np.asarray(scores, dtype=np.float64))
        best_references.append(pad_or_truncate_reference(refs[int(np.argmax(scores))]))

    difference = pred - gt
    distance = np.linalg.norm(difference, axis=-1)
    per_sample = {
        "ADE": distance.mean(axis=1),
        "FDE": distance[:, -1],
        "action_l1": np.abs(difference).mean(axis=(1, 2)),
        "action_l2": (difference ** 2).mean(axis=(1, 2)),
    }
    for seconds in (1, 3, 5):
        count = seconds * 4
        per_sample[f"ADE@{seconds}s"] = distance[:, :count].mean(axis=1)
        per_sample[f"FDE@{seconds}s"] = distance[:, count - 1]
    best_distance = np.linalg.norm(pred - np.stack(best_references), axis=-1)
    per_sample["ADE_vs_best_pref"] = best_distance.mean(axis=1)
    per_sample["ADE_vs_best_pref@3s"] = best_distance[:, :12].mean(axis=1)

    result = get_rater_feedback_score(
        inference_trajectories=pred[:, None, :, :],
        inference_probs=np.ones((len(pred), 1), dtype=np.float64),
        rater_specified_trajectories=reference_batches,
        rater_feedback_labels=score_batches,
        init_speed=np.linalg.norm(velocity, axis=1),
        **RFS_SETTINGS,
    )
    per_sample["RFS"] = np.asarray(result["rater_feedback_score"], dtype=np.float64)
    per_sample["RFS_within_trust_pct"] = (
        np.asarray(result["is_fully_within_trust_region"]).any(axis=-1).astype(np.float64) * 100
    )
    if not all(np.isfinite(per_sample[name]).all() for name in SCORE_NAMES):
        raise ValueError("Nonfinite Waymo evaluation metrics")
    metrics = {name: float(per_sample[name].mean()) for name in SCORE_NAMES}
    metrics.update(num_samples=len(pred), num_pref_evaluated=len(pred))
    return metrics, per_sample
