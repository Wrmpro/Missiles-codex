"""Mouse controller for the game's virtual joystick."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import pyautogui

import config


class PCController:
    """Control the Google Play Games virtual joystick with the mouse.

    The controller applies light smoothing and a deadzone so the joystick
    vector is stable (no per-frame jitter) while still turning quickly, and
    avoids issuing redundant ``moveTo`` calls when the target barely changes to
    reduce CPU/input churn. It supports *continuous motion*: a small non-zero
    vector keeps the joystick engaged rather than releasing every frame.
    """

    def __init__(self, calibration_path: str = config.CALIBRATION_FILE) -> None:
        self.center: Tuple[int, int] = config.JOYSTICK_CENTER
        self.max_radius = config.JOYSTICK_MAX_RADIUS
        self._is_holding = False
        self._smoothed: Tuple[float, float] = (0.0, 0.0)
        self._last_target: Optional[Tuple[float, float]] = None
        path = Path(calibration_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.center = tuple(data.get("joystick_center", self.center))  # type: ignore[assignment]
                self.max_radius = int(data.get("joystick_radius", self.max_radius))
            except (json.JSONDecodeError, OSError, ValueError):
                pass

    def update_calibration(self, center: Tuple[int, int], radius: int) -> None:
        """Update joystick geometry after a (re)calibration without restarting."""

        if radius and radius > 0:
            self.center = (int(center[0]), int(center[1]))
            self.max_radius = int(radius)

    def send(self, dx: float, dy: float) -> None:
        """Move the held joystick toward normalized direction dx, dy."""

        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))

        # Exponential smoothing for jitter-free, still-responsive control.
        alpha = config.CONTROLLER_SMOOTHING
        sx = alpha * dx + (1.0 - alpha) * self._smoothed[0]
        sy = alpha * dy + (1.0 - alpha) * self._smoothed[1]
        self._smoothed = (sx, sy)

        magnitude = math.hypot(sx, sy)
        if magnitude < config.CONTROLLER_DEADZONE:
            # Inside the deadzone we release rather than holding a null vector.
            self.release()
            return

        target_x = self.center[0] + sx * self.max_radius
        target_y = self.center[1] + sy * self.max_radius
        try:
            if not self._is_holding:
                pyautogui.moveTo(*self.center, duration=0)
                pyautogui.mouseDown()
                self._is_holding = True
                self._last_target = None
            # Skip redundant tiny movements.
            if self._last_target is None or math.hypot(
                target_x - self._last_target[0], target_y - self._last_target[1]
            ) >= 1.5:
                pyautogui.moveTo(target_x, target_y, duration=0)
                self._last_target = (target_x, target_y)
        except pyautogui.FailSafeException:
            self._is_holding = False
            raise
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        """Release the virtual joystick safely."""

        self._smoothed = (0.0, 0.0)
        self._last_target = None
        if self._is_holding:
            try:
                pyautogui.moveTo(*self.center, duration=0)
            finally:
                pyautogui.mouseUp()
            self._is_holding = False
        else:
            pyautogui.mouseUp()
