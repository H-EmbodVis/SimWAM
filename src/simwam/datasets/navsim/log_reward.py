"""DreamerAD-style log aggregation, shared by the v1 and v2 reward backends."""

from math import isfinite, log
from typing import Mapping, Sequence


class LogMetricAggregator:
    """Sum log compliance terms, then add log of the weighted additive sum.

    Clamp each metric BEFORE aggregation. Do not normalize the additive weights
    or clip the result to [0, 1]. Metrics are supplied by the NavSim scorer rather
    than DreamerAD's learned reward model.
    """

    def __init__(
        self,
        multiplicative_keys: Sequence[str],
        weighted_metrics: Mapping[str, float],
        epsilon: float = 1e-2,
    ):
        self.multiplicative_keys = tuple(multiplicative_keys)
        self.weighted_metrics = {name: float(weight) for name, weight in weighted_metrics.items()}
        self.metric_names = (*self.multiplicative_keys, *self.weighted_metrics)
        self.epsilon = float(epsilon)
        if not isfinite(self.epsilon) or not 0.0 < self.epsilon < 1.0:
            raise ValueError("log_epsilon must be finite and strictly between 0 and 1.")
        if any(not isfinite(weight) or weight < 0.0 for weight in self.weighted_metrics.values()):
            raise ValueError("Log reward weights must be finite and non-negative.")
        weight_sum = sum(self.weighted_metrics.values())
        if not isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("Log reward additive weights must have a positive finite sum.")
        # A failure must not receive 0: valid log rewards can be negative.
        self.minimum = (len(self.multiplicative_keys) + 1) * log(self.epsilon) + log(weight_sum)
        self.maximum = log(weight_sum)

    def __call__(self, terms: Mapping[str, float]) -> float:
        values = {}
        for name in self.metric_names:
            value = float(terms[name])
            if not isfinite(value):
                raise ValueError(f"Log reward metric {name!r} is not finite.")
            values[name] = max(value, self.epsilon)
        compliance = sum(log(values[name]) for name in self.multiplicative_keys)
        additive = sum(weight * values[name] for name, weight in self.weighted_metrics.items())
        return compliance + log(additive)
