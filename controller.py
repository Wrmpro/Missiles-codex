"""Mouse controller for the game's virtual joystick."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pyautogui

import config


class PCController:
    """Control the Google Play Games virtual joystick with the mouse."""

    def __init__(self, calibration_path: str = config.CALIBRATION_FILE) -> None:
        self.center: Tuple[int, int] = config.JOYSTICK_CENTER
        self.max_radius = config.JOYSTICK_MAX_RADIUS
        self._is_holding = False
        path = Path(calibration_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.center = tuple(data.get("joystick_center", self.center))  # type: ignore[assignment]
            self.max_radius = int(data.get("joystick_radius", self.max_radius))

    def send(self, dx: float, dy: float) -> None:
        """Move the held joystick toward normalized direction dx, dy."""

        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))
        target_x = self.center[0] + dx * self.max_radius
        target_y = self.center[1] + dy * self.max_radius
        try:
            if not self._is_holding:
                pyautogui.moveTo(*self.center, duration=0)
                pyautogui.mouseDown()
                self._is_holding = True
            pyautogui.moveTo(target_x, target_y, duration=0)
            if abs(dx) < 0.05 and abs(dy) < 0.05:
                self.release()
        except pyautogui.FailSafeException:
            self._is_holding = False
            raise
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        """Release the virtual joystick safely."""

        if self._is_holding:
            pyautogui.moveTo(*self.center, duration=0)
            pyautogui.mouseUp()
            self._is_holding = False
        else:
            pyautogui.mouseUp()
