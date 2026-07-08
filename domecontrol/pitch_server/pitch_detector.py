"""
Pitch detection with automatic fallback: crepe -> aubio -> YIN (numpy).

Each backend exposes the same interface:
    detect(audio: np.ndarray, sample_rate: int) -> tuple[float, float]
        Returns (frequency_hz, confidence).  frequency_hz == 0 means no pitch.
"""

import logging
import math

import numpy as np

from .note_utils import MIN_FREQ, MAX_FREQ

logger = logging.getLogger(__name__)


# ============================================
# Backend: CREPE  (deep-learning pitch tracker)
# =============================================

def _detect_crepe(audio: np.ndarray, sr: int) -> tuple[float, float]:
    import crepe  # type: ignore[import-untyped]

    # crepe expects float32 in [-1, 1] and a 1-D array.
    _, freq_arr, conf_arr, _ = crepe.predict(
        audio.astype(np.float32), sr,
        model_capacity="tiny",  # fast enough for real-time
        viterbi=False,
        step_size=int(1000 / 10),  # ~10 predictions/sec
    )
    # Take the prediction with the highest confidence.
    best = int(np.argmax(conf_arr))
    return float(freq_arr[best]), float(conf_arr[best])


# ============================================
# Backend: aubio
# ============================================

# aubio.pitch objects keyed on (buf_size, sample_rate) — constructing one per
# call discards the tracker's cross-frame state and wastes allocation.
_aubio_detectors: dict[tuple[int, int], object] = {}


def _detect_aubio(audio: np.ndarray, sr: int) -> tuple[float, float]:
    import aubio  # type: ignore[import-untyped]

    buf_size = len(audio)
    pitch_o = _aubio_detectors.get((buf_size, sr))
    if pitch_o is None:
        # "yin", not "yinfft": yinfft requires power-of-two buffer sizes on
        # Ooura-FFT builds (the default pip install), and its get_confidence()
        # always returns 0.0 in aubio 0.4.9.
        pitch_o = aubio.pitch("yin", buf_size, buf_size, sr)
        pitch_o.set_unit("Hz")
        pitch_o.set_silence(-40)
        pitch_o.set_tolerance(0.8)
        _aubio_detectors[(buf_size, sr)] = pitch_o

    samples = audio.astype(np.float32)
    freq = float(pitch_o(samples)[0])
    conf = float(pitch_o.get_confidence())
    return freq, conf


# ============================================
# Backend: YIN  (pure-numpy autocorrelation)
# ============================================

def _yin_pitch(audio: np.ndarray, sr: int,
               fmin: float = MIN_FREQ, fmax: float = MAX_FREQ,
               threshold: float = 0.15) -> tuple[float, float]:
    """
    YIN fundamental-frequency estimator (de Cheveigné & Kawahara, 2002).

    Returns (frequency, confidence).  confidence ∈ [0, 1].
    """
    n = len(audio)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(n // 2, int(sr / fmin))

    if tau_max <= tau_min:
        return 0.0, 0.0

    # Step 1 & 2: difference function
    x = audio.astype(np.float64)
    diff = np.zeros(tau_max)
    for tau in range(1, tau_max):
        diff[tau] = np.sum((x[:n - tau] - x[tau:n]) ** 2)

    # Step 3: cumulative mean normalised difference
    cmnd = np.ones(tau_max)
    running_sum = 0.0
    for tau in range(1, tau_max):
        running_sum += diff[tau]
        if running_sum == 0:
            cmnd[tau] = 1.0
        else:
            cmnd[tau] = diff[tau] * tau / running_sum

    # Step 4: absolute threshold — find first dip below threshold
    best_tau = 0
    for tau in range(tau_min, tau_max - 1):
        if cmnd[tau] < threshold:
            # Walk to the local minimum.
            while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            best_tau = tau
            break

    if best_tau == 0:
        # No dip found — pick global minimum in range as a last resort.
        best_tau = int(tau_min + np.argmin(cmnd[tau_min:tau_max]))
        if cmnd[best_tau] > 0.5:
            return 0.0, 0.0  # too uncertain

    freq = sr / best_tau
    # Confidence: invert the cmnd value (lower cmnd = more periodic).
    conf = max(0.0, min(1.0, 1.0 - cmnd[best_tau]))
    return float(freq), float(conf)


def _detect_yin(audio: np.ndarray, sr: int) -> tuple[float, float]:
    return _yin_pitch(audio, sr)


# ===================================
# Unified detector with fallback
# ===================================

_BACKENDS = {
    "crepe": _detect_crepe,
    "aubio": _detect_aubio,
    "yin": _detect_yin,
}

_FALLBACK_ORDER = ["crepe", "aubio", "yin"]


def select_backend(method: str = "auto"):
    """
    Return a callable (audio, sr) -> (freq, conf) for the chosen method.

    With method="auto", tries crepe, then aubio, then yin.
    """
    if method != "auto":
        if method not in _BACKENDS:
            raise ValueError(f"Unknown detection method: {method!r}")
        # Verify the import works.
        try:
            _BACKENDS[method](np.zeros(1024, dtype=np.float32), 44100)
        except ImportError:
            raise ImportError(f"Backend {method!r} is not installed")
        logger.info("Pitch detection backend: %s (explicitly selected)", method)
        return _BACKENDS[method]

    for name in _FALLBACK_ORDER:
        try:
            # Quick smoke test — import check.
            _BACKENDS[name](np.zeros(1024, dtype=np.float32), 44100)
            logger.info("Pitch detection backend: %s", name)
            return _BACKENDS[name]
        except ImportError:
            logger.debug("Backend %s not available, trying next", name)
        except Exception:
            # Some backends may error on a zero buffer — that's fine,
            # the import succeeded.
            logger.info("Pitch detection backend: %s", name)
            return _BACKENDS[name]

    # Should never happen — yin is pure-numpy.
    logger.info("Pitch detection backend: yin (final fallback)")
    return _detect_yin
