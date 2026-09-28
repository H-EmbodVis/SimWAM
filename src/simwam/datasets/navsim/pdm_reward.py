"""NavSim PDM closed-loop score as a GRPO reward.

Faithful port of recogdrive's `ReCogDriveDiffusionPlanner.reward_fn`
(`recogdrive/navsim/agents/recogdrive/recogdrive_diffusion_planner.py`):
a predicted future trajectory (8 ego-frame poses, x/y/heading) is scored against
a per-token NavSim metric cache via `pdm_score`, returning a scalar in [0, 1].

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

from .log_reward import LogMetricAggregator
from .span_reward import SpanMetricAggregator

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

# What `run_pdm_score.py` actually REPORTS (PDMScorerConfig defaults, pdm_scorer.py:39-42). The
# `progress_weight` / `ttc_weight` / `comfortable_weight` arguments below configure the *reward*, and
# configs/train_grpo.yaml sets them to (10, 5, 2) -- deliberately NOT the reported (5, 5, 2). Reward
# and metric are therefore different objectives, so a reward gain can coexist with a metric loss.
# `last_true_pdms` recomputes the reported metric from the SAME simulation's submetrics (no second
# closed-loop sim) purely as a diagnostic, so divergence shows up during training rather than only at
# the next checkpoint eval. driving_direction_compliance carries weight 0.0 upstream, so it is
# reported but contributes nothing -- this formula was verified to 2e-16 against the official scorer
# on all 12146 navtest tokens.
_METRIC_GATES = ("no_at_fault_collisions", "drivable_area_compliance")
_METRIC_WEIGHTS = {"ego_progress": 5.0, "time_to_collision_within_bound": 5.0, "comfort": 2.0}
# Columns kept PER SAMPLE for the per-sub-metric advantage (`grpo.train.subadv`). The scalar PDMS is a
# lossy projection of these five: it multiplies the two gates into the weighted sum, so a group whose
# progress is saturated at 1.0 for every member still shows PDMS variance from TTC/comfort jitter, and
# a group-relative advantage on the scalar then ranks that jitter as if it were policy signal. Keeping
# the columns costs nothing -- `pdm_score` already computes all of them and they were being dropped.
_SUBMETRIC_COLUMNS = (*_METRIC_GATES, *_METRIC_WEIGHTS)


def _metric_from_terms(terms: Dict[str, object]) -> float:
    """Reported PDMS from one trajectory's submetrics. NaN if a submetric is absent.

    NaN rather than an exception: this only feeds a diagnostic, and a devkit that renames a submetric
    must not be able to take the reward path down with it.
    """
    try:
        gate = 1.0
        for key in _METRIC_GATES:
            gate *= float(terms[key])
        num = sum(w * float(terms[key]) for key, w in _METRIC_WEIGHTS.items())
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return gate * num / sum(_METRIC_WEIGHTS.values())


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
        self._true_pdms: list[float] = []
        self._submetric_rows: list[dict] = []

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

    @property
    def failure_reward(self) -> float:
        """Fallback for missing caches or failed scoring; log variants use their lower bound."""
        return 0.0

    def _reward_from_terms(self, terms: Dict[str, object]) -> float:
        return float(terms["score"])

    def _finalize_reward(self, score: float) -> float:
        return float(np.clip(score, 0.0, 1.0))

    def score(self, abs_poses: np.ndarray, metric_cache: object) -> float:
        """Reward for one [N, 3] ego-frame trajectory; standard PDM is in [0, 1]."""
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
        terms = asdict(result)
        # Same simulation, second read-out: the REPORTED metric, for the divergence diagnostic.
        self._true_pdms.append(_metric_from_terms(terms))
        # Third read-out, also free: the per-sample sub-metric row, kept for the per-sub-metric
        # advantage. NaN for any column this devkit does not report, so a renamed column degrades to
        # "that component is masked" rather than taking the reward path down.
        self._submetric_rows.append(
            {name: float(terms.get(name, float("nan"))) for name in _SUBMETRIC_COLUMNS}
        )
        score = self._reward_from_terms(terms)
        if not np.isfinite(score):
            return self.failure_reward
        return self._finalize_reward(score)

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
            rewards: [M] float tensor. Standard PDM is in [0, 1]; failures use failure_reward.
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

        # Per-call failure accounting (read by the trainer for `step/pdm_fail_frac`). A failure is
        # NOT the same as a genuinely bad trajectory: `pdm_score` can raise on a degenerate /
        # out-of-range trajectory, and the 0.0 fallback then enters the group advantage as if the
        # sample were merely terrible. Worth watching whenever the injected noise is raised.
        self.last_num_total = len(tokens)
        self.last_num_missing = 0  # no metric cache for this token
        self.last_num_failed = 0   # pdm_score raised
        # Per-call, for the same reason as the counters above: `score` appends to these.
        self._true_pdms: list[float] = []
        self._submetric_rows: list[dict] = []

        rewards = []
        nan_row = {name: float("nan") for name in _SUBMETRIC_COLUMNS}
        for i, token in enumerate(tokens):
            metric_cache = metric_cache_map.get(token)
            if metric_cache is None:
                self.last_num_missing += 1
                rewards.append(self.failure_reward)
                # Keep the row axis aligned with `rewards`: a sample that never reached the simulator
                # has no sub-metrics, and NaN makes every component mask itself for that sample.
                self._submetric_rows.append(dict(nan_row))
                continue
            try:
                rewards.append(self.score(poses_np[i], metric_cache))
            except Exception as exc:  # PDM can fail on degenerate trajectories.
                self.last_num_failed += 1
                logger.warning("PDM scoring failed for token %s: %s", token, exc)
                rewards.append(self.failure_reward)
                if len(self._submetric_rows) < len(rewards):
                    self._submetric_rows.append(dict(nan_row))
        return torch.tensor(rewards, device=device, dtype=torch.float32)

    @property
    def submetric_columns(self) -> tuple:
        """Names of the per-sample sub-metric columns, in the order `last_submetric_matrix` returns."""
        return _SUBMETRIC_COLUMNS

    def last_submetric_matrix(self, device=None) -> torch.Tensor:
        """`[M, 5]` per-sample sub-metrics from the last `score_batch`, aligned with its rewards.

        Column order is `submetric_columns` = (no_at_fault_collisions, drivable_area_compliance,
        ego_progress, time_to_collision_within_bound, comfort). Empty tensor before the first call.
        NOTE this is deliberately NOT called `last_submetrics`: the EPDMS reward uses that name for
        per-batch MEANS, which the trainer logs. This one is per SAMPLE and drives the advantage.
        """
        rows = getattr(self, "_submetric_rows", [])
        if not rows:
            return torch.zeros((0, len(_SUBMETRIC_COLUMNS)), device=device, dtype=torch.float32)
        return torch.tensor(
            [[row[name] for name in _SUBMETRIC_COLUMNS] for row in rows],
            device=device,
            dtype=torch.float32,
        )

    @property
    def last_fail_frac(self) -> float:
        """Fraction of the last `score_batch` whose PDM score came from a failure (raise or no
        metric cache) rather than an actual simulation. 0.0 before the first call."""
        total = int(getattr(self, "last_num_total", 0))
        if total <= 0:
            return 0.0
        bad = int(getattr(self, "last_num_failed", 0)) + int(getattr(self, "last_num_missing", 0))
        return float(bad) / float(total)

    @property
    def last_true_pdms(self) -> float:
        """Mean REPORTED PDMS over the last `score_batch`; NaN before the first call.

        The reward uses `progress_weight` / `ttc_weight` / `comfortable_weight`, which default to
        (10, 5, 2) while the reported metric is fixed at (5, 5, 2). This is therefore the only
        in-training signal that the reward is rising while the metric is not -- without it, that only
        becomes visible at the next checkpoint eval, ~500 optimizer steps later.
        """
        vals = [v for v in getattr(self, "_true_pdms", []) if np.isfinite(v)]
        if not vals:
            return float("nan")
        return float(np.mean(vals))


class NavSimLogPDMReward(NavSimPDMReward):
    """Log PDMS with NC/DAC gates and the existing training weights EP/TTC/C=10/5/2.

    DDC remains excluded. The reported `last_true_pdms` diagnostic still uses the
    original v1 metric weights 5/5/2 and is not log-transformed.
    """

    def __init__(
        self,
        metric_cache_path: str,
        progress_weight: float = 10.0,
        ttc_weight: float = 5.0,
        comfortable_weight: float = 2.0,
        log_epsilon: float = 1e-2,
        **kwargs,
    ):
        self._log_aggregation = LogMetricAggregator(
            _METRIC_GATES,
            {"ego_progress": progress_weight, "time_to_collision_within_bound": ttc_weight,
             "comfort": comfortable_weight},
            epsilon=log_epsilon,
        )
        super().__init__(
            metric_cache_path,
            progress_weight=progress_weight,
            ttc_weight=ttc_weight,
            comfortable_weight=comfortable_weight,
            **kwargs,
        )
        logger.info("Log PDM reward: epsilon=%g weights=%s range=[%.6f, %.6f]",
                    log_epsilon, self._log_aggregation.weighted_metrics,
                    self.failure_reward, self._log_aggregation.maximum)

    @property
    def failure_reward(self) -> float:
        return self._log_aggregation.minimum

    def _reward_from_terms(self, terms: Dict[str, object]) -> float:
        return self._log_aggregation(terms)

    def _finalize_reward(self, score: float) -> float:
        return float(score)


class NavSimSpanPDMReward(NavSimPDMReward):
    """Curious-VLA Eq. 10: NC * DAC * weighted_mean(1-(1-m)**gamma).

    Paper defaults are EP/TTC/C weights 5/5/2 and exponents 0.5/0.5/1.
    DDC stays excluded; the official PDMS diagnostic uses untransformed metrics.
    """

    def __init__(
        self,
        metric_cache_path: str,
        progress_weight: float = 5.0,
        ttc_weight: float = 5.0,
        comfortable_weight: float = 2.0,
        progress_exponent: float = 0.5,
        ttc_exponent: float = 0.5,
        comfortable_exponent: float = 1.0,
        **kwargs,
    ):
        self._span_aggregation = SpanMetricAggregator(
            _METRIC_GATES,
            {"ego_progress": progress_weight, "time_to_collision_within_bound": ttc_weight,
             "comfort": comfortable_weight},
            {"ego_progress": progress_exponent, "time_to_collision_within_bound": ttc_exponent,
             "comfort": comfortable_exponent},
        )
        super().__init__(
            metric_cache_path,
            progress_weight=progress_weight,
            ttc_weight=ttc_weight,
            comfortable_weight=comfortable_weight,
            **kwargs,
        )
        logger.info("Span PDM reward: weights=%s exponents=%s",
                    self._span_aggregation.weighted_metrics, self._span_aggregation.focal_exponents)

    def _reward_from_terms(self, terms: Dict[str, object]) -> float:
        return self._span_aggregation(terms)
