"""
Audio capture and pitch-processing loop.

Runs in a background thread, captures microphone audio via sounddevice,
feeds it through the selected pitch detector, maintains a rolling buffer
for stabilisation, and updates shared pitchState.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from typing import Callable, Optional, Tuple

import numpy as np
import sounddevice as sd

from .config import PitchConfig
from .note_utils import MIN_FREQ, MAX_FREQ, snap_to_note
from .pitch_detector import select_backend
from .state import PitchReading, PitchState

logger = logging.getLogger(__name__)


def list_audio_devices() -> None:
    """Print all available audio devices to stdout."""
    print("Available audio devices:")
    print(sd.query_devices())
    print()


def _rms_to_db(rms: float) -> float:
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms)


def audio_loop(
    state: PitchState,
    config: PitchConfig,
    on_change: Optional[Callable] = None,
) -> None:
    """
    Continuously capture audio, detect pitch, and update *state*.

    Parameters
    ----------
    state : PitchState
        Thread-safe container written to on each analysis frame.
    config : PitchConfig
        Audio / detection settings.
    on_change : callable, optional
        Called with a PitchReading dict whenever the detected note changes.
        Intended for publishing SSE events.
    """
    list_audio_devices()

    sr = config.sample_rate
    block_size = sr // config.update_rate_hz  # samples per analysis window

    # Select pitch detection backend.
    detect_fn = select_backend(config.detection_method)

    # Rolling buffer of recent (note, confidence) pairs for stabilisation.
    history: collections.deque[tuple[str | None, float]] = collections.deque(
        maxlen=config.rolling_buffer_frames,
    )

    prev_note: str | None = None
    last_log_time = 0.0

    device = config.device  # None means system default

    logger.info(
        "Starting audio capture: device=%s  sr=%d  block=%d  update_rate=%d Hz",
        device, sr, block_size, config.update_rate_hz,
    )

    try:
        with sd.InputStream(
            device=device,
            channels=1,
            samplerate=sr,
            blocksize=block_size,
            dtype="float32",
        ) as stream:
            while True:
                audio_block, overflowed = stream.read(block_size)
                if overflowed:
                    logger.warning("Audio input overflowed")

                mono = audio_block[:, 0]  # shape: (block_size,)

                # Check noise floor.
                rms = float(np.sqrt(np.mean(mono ** 2)))
                level_db = _rms_to_db(rms)

                if level_db < config.noise_floor_db:
                    # Silence.
                    history.append((None, 0.0))
                    reading = PitchReading(timestamp=time.time())
                else:
                    freq, conf = detect_fn(mono, sr)

                    if freq < MIN_FREQ or freq > MAX_FREQ or conf < 0.1:
                        history.append((None, 0.0))
                        reading = PitchReading(timestamp=time.time())
                    else:
                        note_name, _, cents = snap_to_note(freq)
                        history.append((note_name, conf))
                        reading = PitchReading(
                            note=note_name,
                            frequency=freq,
                            confidence=conf,
                            cents_offset=cents,
                            timestamp=time.time(),
                        )

                # Stabilise: pick the most common recent note, weighted by
                # confidence.
                stabilised = _stabilise(history)
                if stabilised is not None:
                    # Override note with stabilised value but keep freq/cents
                    # from the current frame.
                    reading = PitchReading(
                        note=stabilised[0],
                        frequency=reading.frequency,
                        confidence=stabilised[1],
                        cents_offset=reading.cents_offset,
                        timestamp=reading.timestamp,
                    )

                state.update(reading)

                # Publish SSE event when the note changes.
                current_note = reading.note
                if current_note != prev_note and on_change is not None:
                    on_change(reading.to_dict())
                prev_note = current_note

                # Log a summary ~once per second.
                now = time.monotonic()
                if now - last_log_time >= 1.0:
                    last_log_time = now
                    logger.info(
                        "pitch: note=%s  freq=%s  conf=%.2f  level=%.1f dB",
                        reading.note,
                        f"{reading.frequency:.1f}" if reading.frequency else "-",
                        reading.confidence,
                        level_db,
                    )

    except Exception:
        logger.exception("Audio loop crashed")
        raise


def _stabilise(
    history: collections.deque[tuple[str | None, float]],
) -> tuple[str, float] | None:
    """
    From a rolling buffer of (note, confidence) pairs, return the
    dominant (note, avg_confidence) weighted by confidence.

    Returns None if all entries are silence.
    """
    if not history:
        return None

    weighted: dict[str, float] = {}
    counts: dict[str, int] = {}
    for note, conf in history:
        if note is None:
            continue
        weighted[note] = weighted.get(note, 0.0) + conf
        counts[note] = counts.get(note, 0) + 1

    if not weighted:
        return None

    best_note = max(weighted, key=weighted.get)  # type: ignore[arg-type]
    avg_conf = weighted[best_note] / counts[best_note]

    # Boost confidence if recent frames agree.
    agreement = counts[best_note] / len(history)
    blended_conf = min(1.0, avg_conf * (0.5 + 0.5 * agreement))
    return best_note, blended_conf
