"""
Colored light detection from a USB camera in a dark room.

Captures 1080p frames, finds bright spots (blobs), and classifies each
blob's color via HSV hue. Returns a sorted list of detected color names
for each frame.
"""

from __future__ import annotations

import cv2
import numpy as np

# Brightness threshold — pixels above this value (0-255 grayscale) are
# considered "lit".  In a dark room with only light-wand / phone-screen
# sources this should be pretty reliable!
BRIGHTNESS_THRESH = 100

# Minimum contour area in pixels to count as a light (filters noise.)
MIN_BLOB_AREA = 15

# Maximum contour area — if a blob is huge it's probably ambient light
# or a reflection, not a handheld light dot.
MAX_BLOB_AREA = 8000

# Saturation threshold: below this we call it "white" instead of
# trying to read a specific hue.
WHITE_SAT_THRESH = 40

# HSV hue ranges mapped to color names.  OpenCV hue is 0-179.
# Each entry is (low_hue, high_hue, name).
# Note: these MAY need to be tuned a little for a room, to accomodate screen
# colorspaces, brightness, etc. if going with phones.
HUE_RANGES = [
    (0, 10, "red"),
    (11, 25, "orange"),
    (26, 34, "yellow"),
    (35, 80, "green"),
    (81, 100, "cyan"),
    (101, 130, "blue"),
    (131, 145, "purple"),
    (146, 165, "magenta"),
    (166, 179, "red"),  # red wraps around
]


def classify_hue(hue: int, saturation: int) -> str:
    """Return a color name for an OpenCV HSV hue value (0-179)."""
    if saturation < WHITE_SAT_THRESH:
        return "white"
    for low, high, name in HUE_RANGES:
        if low <= hue <= high:
            return name
    return "white"


def detect_colors(frame: np.ndarray) -> list[str]:
    """
    Given a BGR frame, return a *sorted* list of detected color names.

    Each distinct bright blob produces one entry; if two green lights are
    visible the list will contain "green" twice.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate bright spots.
    _, mask = cv2.threshold(gray, BRIGHTNESS_THRESH, 255, cv2.THRESH_BINARY)

    # Optional: slight blur + re-threshold to merge nearby active pixels
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colors: list[str] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
            continue

        # Build a mask for just this and compute mean HSV
        blob_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        mean_hsv = cv2.mean(hsv, mask=blob_mask)  # (H, S, V, _)

        color_name = classify_hue(int(mean_hsv[0]), int(mean_hsv[1]))
        colors.append(color_name)

    colors.sort()
    return colors


class Camera:
    """Thin wrapper around cv2.VideoCapture for 1080p USB camera."""

    def __init__(self, device: int = 0):
        self.cap = cv2.VideoCapture(device)
        # Request 1080p.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def read_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
