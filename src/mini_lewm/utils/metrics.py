from __future__ import annotations

from collections import defaultdict


class MetricTracker:
    def __init__(self) -> None:
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self._sums[key] += float(value)
            self._counts[key] += 1

    def averages(self) -> dict[str, float]:
        return {
            key: self._sums[key] / max(self._counts[key], 1)
            for key in self._sums
        }
