"""Official Waymo RFS as one scalar reward per GRPO candidate trajectory."""

from __future__ import annotations

import numpy as np
import torch

from ._vendor.rater_feedback_utils import get_rater_feedback_score
from .metrics import RFS_SETTINGS
from .waymo_dataset import WaymoVideoDataset


class WaymoRFSReward:
    """Score physical-unit [M,20,2] predictions using token-matched references.

    The trainer has already inverted the IL action normalization. Repeated tokens
    are intentional: each of the G proposals receives its own RFS in [0,10].
    There is no group averaging, division by 10, heading, or interpolation here.
    """

    name = "rfs"

    def __init__(self, dataset: WaymoVideoDataset):
        if not isinstance(dataset, WaymoVideoDataset):
            raise TypeError("Waymo RFS requires WaymoVideoDataset")
        self._references = {}
        self._scores = {}
        self._speeds = {}
        for index, token in enumerate(dataset._tokens):
            token = str(token)
            references = dataset._references[index]
            if not references:
                raise ValueError(f"Waymo RFS sample lacks scored references: {token}")
            self._references[token] = [
                np.column_stack([r["pos_x"], r["pos_y"]]).astype(np.float64)
                for r in references
            ]
            self._scores[token] = np.asarray([r["preference_score"] for r in references], dtype=np.float64)
            self._speeds[token] = float(np.linalg.norm(dataset._states[index, :2].astype(np.float64)))
        if not self._references:
            raise ValueError("Waymo RFS dataset is empty")
        self.last_num_total = 0
        self.last_within_trust_pct = 0.0

    def available(self, token: str) -> bool:
        return str(token) in self._references

    @property
    def last_fail_frac(self) -> float:
        # Invalid input is an error, never an artificial zero-reward candidate.
        return 0.0

    def score_batch(self, abs_poses, tokens, device=None) -> torch.Tensor:
        if isinstance(abs_poses, torch.Tensor):
            if device is None:
                device = abs_poses.device
            trajectories = abs_poses.detach().to(device="cpu", dtype=torch.float64).numpy()
        else:
            trajectories = np.asarray(abs_poses, dtype=np.float64)
        if device is None:
            device = torch.device("cpu")
        tokens = [str(token) for token in tokens]
        if trajectories.shape != (len(tokens), 20, 2) or not tokens:
            raise ValueError("RFS expects one token per native [M,20,2] XY trajectory")
        if not np.isfinite(trajectories).all():
            raise ValueError("RFS predictions contain nonfinite XY values")
        missing = sorted(set(tokens) - self._references.keys())
        if missing:
            raise KeyError(f"Missing RFS reference payload for {len(missing)} token(s): {missing[:3]}")
        self.last_num_total = len(tokens)
        result = get_rater_feedback_score(
            inference_trajectories=trajectories[:, None],
            inference_probs=np.ones((len(tokens), 1), dtype=np.float64),
            rater_specified_trajectories=[self._references[token] for token in tokens],
            rater_feedback_labels=[self._scores[token] for token in tokens],
            init_speed=np.asarray([self._speeds[token] for token in tokens], dtype=np.float64),
            **RFS_SETTINGS,
        )
        rewards = np.asarray(result["rater_feedback_score"], dtype=np.float64)
        if rewards.shape != (len(tokens),) or not np.isfinite(rewards).all():
            raise ValueError("Official RFS returned invalid rewards")
        self.last_within_trust_pct = float(
            result["is_fully_within_trust_region"].any(axis=-1).mean() * 100
        )
        return torch.as_tensor(rewards, device=device, dtype=torch.float32)
