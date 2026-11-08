"""Lightweight per-request tracing: record each pipeline step and how long it took."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    name: str
    duration_ms: float
    details: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """Ordered record of the steps a request went through (returned by the API)."""

    steps: list[TraceStep] = Field(default_factory=list)

    @contextmanager
    def step(self, name: str, **details: Any) -> Generator[dict[str, Any]]:
        """Time a block; the yielded dict can be filled with details as it runs."""
        info: dict[str, Any] = dict(details)
        start = time.perf_counter()
        try:
            yield info
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.steps.append(
                TraceStep(name=name, duration_ms=round(elapsed, 1), details=info)
            )

    @property
    def total_ms(self) -> float:
        return round(sum(s.duration_ms for s in self.steps), 1)
