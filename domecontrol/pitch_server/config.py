"""
Configuration loading — merges config.yaml with CLI overrides.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


@dataclass
class PitchConfig:
    device: int | str | None = None  # None = system default
    sample_rate: int = 44100
    update_rate_hz: int = 10
    noise_floor_db: float = -40.0
    rolling_buffer_frames: int = 5
    detection_method: str = "auto"  # "crepe", "aubio", "yin", or "auto"


def load_config(path: str | None = None) -> PitchConfig:
    """Load config from YAML file, falling back to defaults."""
    cfg = PitchConfig()
    file_path = path or DEFAULT_CONFIG_PATH

    if os.path.isfile(file_path):
        try:
            with open(file_path) as f:
                data = yaml.safe_load(f) or {}
            for key in ("device", "sample_rate", "update_rate_hz",
                        "noise_floor_db", "rolling_buffer_frames",
                        "detection_method"):
                if key in data:
                    setattr(cfg, key, data[key])
            logger.info("Loaded config from %s", file_path)
        except Exception:
            logger.warning("Failed to read %s, using defaults", file_path, exc_info=True)
    else:
        logger.info("No config file at %s, using defaults", file_path)

    return cfg
