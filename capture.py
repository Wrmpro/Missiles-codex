"""Fast screen capture for the calibrated game window."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

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
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                monitor = data.get("monitor")
                if isinstance(monitor, dict):
                    self.monitor = monitor
            except (json.JSONDecodeError, OSError):
                pass

    @property
    def capture_ms(self) -> float:
        """Return the duration of the previous capture in milliseconds."""

        return self._capture_ms

    def set_monitor(self, monitor: Dict[str, int]) -> None:
        """Update the capture rectangle (e.g. after the window moved)."""

        if not monitor:
            return
        required = {"top", "left", "width", "height"}
        if required.issubset(monitor) and monitor["width"] > 0 and monitor["height"] > 0:
            self.monitor = dict(monitor)

    def grab(self, region: Optional[Dict[str, int]] = None) -> np.ndarray:
        """Capture a frame and return it in OpenCV BGR format.

        ``region`` overrides the configured monitor for one-off grabs (used for
        cropping to the game canvas without mutating the capture state).
        """

        start = time.perf_counter()
        raw = np.array(self._sct.grab(region or self.monitor))
        self._capture_ms = (time.perf_counter() - start) * 1000.0
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
