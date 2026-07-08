"""
Thread-safe shared state for the current pitch reading.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class PitchReading:
    note: str | None = None         # e.g. "A4", None if silence
    frequency: float | None = None  # raw Hz, None if silence
    confidence: float = 0.0         # 0.0 – 1.0
    cents_offset: int = 0           # cents sharp (+) or flat (-)
    timestamp: float = 0.0          # unix time of detection

    def to_dict(self) -> dict:
        return {
            "note": self.note,
            "frequency": round(self.frequency, 2) if self.frequency is not None else 0.0,
            "confidence": round(self.confidence, 3),
            "cents_offset": self.cents_offset,
            "timestamp": round(self.timestamp, 3),
        }


class PitchState:
    """Holds the latest pitch reading behind a lock."""

    def __init__(self) -> None:
        self._reading = PitchReading()
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

    def update(self, reading: PitchReading) -> None:
        with self._lock:
            self._reading = reading

    def get(self) -> PitchReading:
        with self._lock:
            return self._reading

    @property
    def uptime(self) -> float:
        return time.monotonic() - self._start_time
