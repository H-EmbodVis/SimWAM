"""NavSim PDM closed-loop score as a GRPO reward.

A predicted future trajectory (8 ego-frame poses, x/y/heading) is scored against a
per-token NAVSIM metric cache via `pdm_score`, returning a scalar in [0, 1].

The model trajectory is the 8-pose @ 0.5s sequence (`TrajectorySampling(num_poses=8,
interval_length=0.5)`), while the simulator/scorer run at 40 poses @ 0.1s -- `pdm_score`
interpolates internally. The PDM poses must be expressed in the *current ego frame*
(rear-axle relative), which is exactly what `NavSimVideoDataset.denormalize_action`
returns.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from simwam.utils.logging_config import get_logger

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = get_logger(__name__)


class NavSimPDMReward:
    """Compute NavSim PDM closed-loop scores for predicted ego-frame trajectories."""

    def __init__(
        self,
        metric_cache_path: str,
        proposal_time_horizon: float = 4.0,
        proposal_interval: float = 0.1,
        model_trajectory_interval: float = 0.5,
        progress_weight: float = 10.0,
        ttc_weight: float = 5.0,
        comfortable_weight: float = 2.0,
    ):
        cache_path = Path(metric_cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"NavSim metric cache path does not exist: {cache_path}. "
                "Set `grpo.reward.metric_cache_path` / NAVSIM_METRIC_CACHE_PATH."
            )
        self.metric_cache_loader = MetricCacheLoader(cache_path)
        self.model_trajectory_interval = float(model_trajectory_interval)

        # recogdrive GRPO uses 40 poses @ 0.1s for the closed-loop proposal sampling.
        proposal_sampling = TrajectorySampling(
            time_horizon=float(proposal_time_horizon),
            interval_length=float(proposal_interval),
        )
        self.simulator = PDMSimulator(proposal_sampling)
        self.scorer = PDMScorer(
            proposal_sampling,
            PDMScorerConfig(
                progress_weight=float(progress_weight),
                ttc_weight=float(ttc_weight),
                comfortable_weight=float(comfortable_weight),
            ),
        )
        logger.info(
            "Initialized NavSimPDMReward: cache=%s tokens=%d proposal=%.1fs@%.2fs model_traj_interval=%.2fs",
            str(cache_path),
            len(self.metric_cache_loader),
            float(proposal_time_horizon),
            float(proposal_interval),
            self.model_trajectory_interval,
        )

    def available(self, token: str) -> bool:
        """Whether a metric cache exists for this token (guard before scoring)."""
        return token in self.metric_cache_loader.metric_cache_paths

    def prefetch(self, tokens: Iterable[str]) -> Dict[str, object]:
        """Load metric caches for the unique available tokens once (recogdrive pattern)."""
        cache: Dict[str, object] = {}
        for token in set(tokens):
            if self.available(token):
                cache[token] = self.metric_cache_loader.get_from_token(token)
        return cache

    def score(self, abs_poses: np.ndarray, metric_cache: object) -> float:
        """PDM score in [0, 1] for one [N, 3] ego-frame (x, y, heading[rad]) trajectory."""
        poses = np.asarray(abs_poses, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[-1] != 3:
            raise ValueError(f"`abs_poses` must be [N, 3], got shape {tuple(poses.shape)}")
        num_poses = int(poses.shape[0])
        trajectory = Trajectory(
            poses,
            TrajectorySampling(
                num_poses=num_poses,
                interval_length=self.model_trajectory_interval,
            ),
        )
        result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=self.simulator.proposal_sampling,
            simulator=self.simulator,
            scorer=self.scorer,
        )
        score = float(asdict(result)["score"])
        if not np.isfinite(score):
            return 0.0
        return float(np.clip(score, 0.0, 1.0))

    def score_batch(
        self,
        abs_poses: torch.Tensor | np.ndarray,
        tokens: Sequence[str],
        metric_cache_map: Optional[Dict[str, object]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Score a batch of trajectories.

        Args:
            abs_poses: [M, N, 3] ego-frame absolute poses (denormalized).
            tokens: length-M list of scene tokens aligned with `abs_poses`.
            metric_cache_map: optional pre-loaded {token: metric_cache}; built on demand otherwise.
            device: device of the returned reward tensor (defaults to `abs_poses` device or cpu).

        Returns:
            rewards: [M] float tensor in [0, 1]. Tokens without a metric cache score 0.0.
        """
        if isinstance(abs_poses, torch.Tensor):
            if device is None:
                device = abs_poses.device
            poses_np = abs_poses.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            poses_np = np.asarray(abs_poses, dtype=np.float32)
        if device is None:
            device = torch.device("cpu")
        if poses_np.shape[0] != len(tokens):
            raise ValueError(
                f"Batch mismatch: abs_poses M={poses_np.shape[0]} vs tokens={len(tokens)}"
            )

        if metric_cache_map is None:
            metric_cache_map = self.prefetch(tokens)

        rewards = []
        for i, token in enumerate(tokens):
            metric_cache = metric_cache_map.get(token)
            if metric_cache is None:
                rewards.append(0.0)
                continue
            try:
                rewards.append(self.score(poses_np[i], metric_cache))
            except Exception as exc:  # PDM can fail on degenerate trajectories.
                logger.warning("PDM scoring failed for token %s: %s", token, exc)
                rewards.append(0.0)
        return torch.tensor(rewards, device=device, dtype=torch.float32)
