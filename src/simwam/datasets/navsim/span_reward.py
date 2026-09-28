"""Curious-VLA Spanning Driving Reward (arXiv:2603.06049, Eq. 10 / App. A.3)."""

from math import expm1, isfinite, log1p, prod
from typing import Mapping, Sequence


class SpanMetricAggregator:
    """Keep safety gates and replace each additive metric m by 1-(1-m)**gamma.

    Normalize by the sum of the included weights. Metrics must be finite scores
    in [0, 1]; unavailable metrics such as EC are omitted entirely by the caller.
    A zero safety gate stays zero, with no log floor or extra reward terms.
    """

    def __init__(
        self,
        multiplicative_keys: Sequence[str],
        weighted_metrics: Mapping[str, float],
        focal_exponents: Mapping[str, float],
    ):
        self.multiplicative_keys = tuple(multiplicative_keys)
        self.weighted_metrics = {name: float(weight) for name, weight in weighted_metrics.items()}
        self.focal_exponents = {name: float(gamma) for name, gamma in focal_exponents.items()}
        self.metric_names = (*self.multiplicative_keys, *self.weighted_metrics)
        if any(not isfinite(weight) or weight < 0.0 for weight in self.weighted_metrics.values()):
            raise ValueError("Span reward weights must be finite and non-negative.")
        self.weight_sum = sum(self.weighted_metrics.values())
        if not isfinite(self.weight_sum) or self.weight_sum <= 0.0:
            raise ValueError("Span reward additive weights must have a positive finite sum.")
        if self.focal_exponents.keys() != self.weighted_metrics.keys():
            raise ValueError("Span reward exponents must cover exactly the additive metrics.")
        if any(not isfinite(gamma) or gamma <= 0.0 for gamma in self.focal_exponents.values()):
            raise ValueError("Span reward exponents must be finite and strictly positive.")

    @staticmethod
    def _transform(value: float, gamma: float) -> float:
        if gamma == 1.0 or value == 1.0:
            return value
        # Equivalent to 1-(1-value)**gamma without cancellation near zero.
        return -expm1(gamma * log1p(-value))

    def __call__(self, terms: Mapping[str, float]) -> float:
        values = {}
        for name in self.metric_names:
            value = float(terms[name])
            if not isfinite(value):
                raise ValueError(f"Span reward metric {name!r} is not finite.")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Span reward metric {name!r} must be in [0, 1], got {value}.")
            values[name] = value
        compliance = prod(values[name] for name in self.multiplicative_keys)
        additive = sum(
            weight * self._transform(values[name], self.focal_exponents[name])
            for name, weight in self.weighted_metrics.items()
        )
        return compliance * additive / self.weight_sum
