"""
Frequency-to-note mapping and chromatic scale utilities.

Reference: A4 = 440 Hz.  All 12 semitones of the chromatic scale are
supported from C2 (65.41 Hz) to C6 (1046.50 Hz).
"""

import math

A4_FREQ = 440.0
A4_MIDI = 69  # MIDI note number for A4

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Detection range (inclusive)
MIN_FREQ = 65.41   # C2
MAX_FREQ = 1046.50  # C6


def freq_to_midi(freq: float) -> float:
    """Convert a frequency in Hz to a (fractional) MIDI note number."""
    if freq <= 0:
        return 0.0
    return 12.0 * math.log2(freq / A4_FREQ) + A4_MIDI


def midi_to_freq(midi: float) -> float:
    """Convert a MIDI note number (possibly fractional) back to Hz."""
    return A4_FREQ * (2.0 ** ((midi - A4_MIDI) / 12.0))


def midi_to_note_name(midi_int: int) -> str:
    """Return note name with octave, e.g. 'A4', 'C#3'."""
    note = NOTE_NAMES[midi_int % 12]
    octave = (midi_int // 12) - 1
    return f"{note}{octave}"


def snap_to_note(freq: float) -> tuple[str, float, int]:
    """
    Snap a frequency to the nearest chromatic note.

    Returns (note_name, snapped_freq, cents_offset).
    - note_name:     e.g. "A4"
    - snapped_freq:  exact frequency of the snapped note
    - cents_offset:  how many cents the original freq is sharp (+) or flat (-)
    """
    midi_float = freq_to_midi(freq)
    midi_int = round(midi_float)
    snapped_freq = midi_to_freq(midi_int)
    cents = round((midi_float - midi_int) * 100.0)
    name = midi_to_note_name(midi_int)
    return name, snapped_freq, cents
