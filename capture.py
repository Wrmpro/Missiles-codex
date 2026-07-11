"""Fast screen capture for the calibrated game window."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import mss
import numpy as np

import config


class FrameCapture:
    """Capture BGR frames from the calibrated monitor rectangle."""

    def __init__(self, calibration_path: str = config.CALIBRATION_FILE) -> None:
        self._sct = mss.mss()
        self._capture_ms = 0.0
        self.monitor = dict(config.MONITOR_BOUNDS)
        path = Path(calibration_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.monitor = data.get("monitor", self.monitor)

    @property
    def capture_ms(self) -> float:
        """Return the duration of the previous capture in milliseconds."""

        return self._capture_ms

    def grab(self) -> np.ndarray:
        """Capture a frame and return it in OpenCV BGR format."""

        start = time.perf_counter()
        raw = np.array(self._sct.grab(self.monitor))
        self._capture_ms = (time.perf_counter() - start) * 1000.0
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
