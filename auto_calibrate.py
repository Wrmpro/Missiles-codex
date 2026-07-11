"""Main launcher for the Missiles! autonomous pilot."""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import win32con
import win32gui
import win32api

import config
from detector import Detection

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01


IGNORED_TITLES = (
    "File Explorer",
    "Visual Studio",
    "Terminal",
    "PowerShell",
    "Python",
    "VS Code",
)
GAME_TITLE_MARKERS = ("Missiles!", "Hidrain007", "Hidrian007")
SHELL_TITLE_MARKERS = ("Google Play Games",)
ACCEPTED_TITLES = GAME_TITLE_MARKERS


def _is_ignored_title(title: str) -> bool:
    return any(blocked.lower() in title.lower() for blocked in IGNORED_TITLES)


def _is_real_game_title(title: str) -> bool:
    title_lower = title.lower()
    return any(marker.lower() in title_lower for marker in GAME_TITLE_MARKERS)


def _is_shell_title(title: str) -> bool:
    title_lower = title.lower()
    return any(marker.lower() in title_lower for marker in SHELL_TITLE_MARKERS)


def find_game_window() -> Optional[Any]:
    """Find only the real Missiles game window, never the Google Play Games shell."""

    try:
        candidates = []
        for win in gw.getAllWindows():
            title = (win.title or "").strip()
            if not title or win.width <= 200 or win.height <= 200:
                continue
            if _is_ignored_title(title):
                continue
            if _is_real_game_title(title):
                candidates.append(win)
        if candidates:
            # Prefer the largest real game surface. This avoids picking small helper windows.
            win = max(candidates, key=lambda item: item.width * item.height)
            print(f"[window] Found game: {win.title} ({win.width}x{win.height})")
            return win
    except Exception as exc:
        print(f"[window] Search failed: {exc}")
    return None


def find_shell_window() -> Optional[Any]:
    """Find Google Play Games shell only for launch diagnostics, not gameplay."""

    try:
        for win in gw.getAllWindows():
            title = (win.title or "").strip()
            if title and win.width > 200 and win.height > 200 and not _is_ignored_title(title):
                if _is_shell_title(title) and not _is_real_game_title(title):
                    return win
    except Exception:
        pass
    return None


def wait_for_game_window(timeout: float = 45.0) -> Optional[Any]:
    """Wait for the actual Missiles window after the launcher opens Google Play Games."""

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        win = find_game_window()
        if win is not None:
            return win
        shell = find_shell_window()
        if shell is not None:
            print(f"[window] Waiting for real game window; shell is open: {shell.title}")
        time.sleep(1.0)
    return None


def launch_game() -> bool:
    """Launch Missiles! from a shortcut or Windows search."""

    desktop = Path.home() / "Desktop"
    public_desktop = Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "Desktop"
    start_menu = Path(os.environ.get("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs"
    common_start_menu = Path(os.environ.get("PROGRAMDATA", "")) / r"Microsoft\Windows\Start Menu\Programs"
    candidates = [desktop / "Missiles!.lnk", desktop / "Missiles.lnk", public_desktop / "Missiles!.lnk"]
    for root in (start_menu, common_start_menu):
        if root.exists():
            candidates.extend(root.rglob("*Missiles*.lnk"))
    for shortcut in candidates:
        if shortcut.exists():
            print(f"[launch] Opening {shortcut}")
            os.startfile(shortcut)  # type: ignore[attr-defined]
            time.sleep(3)
            return True
    print("[launch] Shortcut not found. Trying Windows search.")
    try:
        pyautogui.hotkey("win")
        time.sleep(0.8)
        pyautogui.write("Missiles!", interval=0.03)
        time.sleep(1.0)
        pyautogui.press("enter")
        time.sleep(3)
        return True
    except Exception as exc:
        print(f"[launch] Windows search launch failed: {exc}")
    return False


def _foreground_title() -> str:
    hwnd = win32gui.GetForegroundWindow()
    return win32gui.GetWindowText(hwnd) if hwnd else ""


def _is_game_title(title: str) -> bool:
    return _is_real_game_title(title)


def is_game_foreground() -> bool:
    """Return True only when the real game window is foreground."""

    return _is_real_game_title(_foreground_title())


def emergency_stop_pressed() -> bool:
    """Return True when the user presses F8."""

    return bool(win32api.GetAsyncKeyState(win32con.VK_F8) & 0x8000)


def focus_window(win: Any) -> bool:
    """Restore and focus the game window, returning True if it becomes foreground."""

    try:
        hwnd = getattr(win, "_hWnd", None)
        if win.isMinimized:
            win.restore()
            time.sleep(0.5)
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_NOTOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass
        try:
            win.activate()
        except Exception:
            pass
        time.sleep(0.5)
        if _is_game_title(_foreground_title()):
            pyautogui.click(win.left + win.width // 2, win.top + min(120, win.height // 4))
            time.sleep(0.5)
            print(f"[window] Foreground: {_foreground_title()}")
            return True
        print(f"[window] Could not focus game. Foreground is: {_foreground_title()!r}")
    except Exception as exc:
        print(f"[window] Focus failed: {exc}")
    return False


def calibrate_from_window(win: Any) -> dict[str, Any]:
    """Save capture and joystick coordinates from the window geometry."""

    left, top, width, height = int(win.left), int(win.top), int(win.width), int(win.height)
    game_rect = (0, 0, width, height)
    try:
        frame = _grab_window_bgr(win)
        game_rect = detect_game_rect(frame)
    except Exception as exc:
        print(f"[calibrate] Could not detect game canvas, using whole window: {exc}")
    gx, gy, gwid, ghei = game_rect
    joystick_center = (left + gx + gwid // 2, top + gy + int(ghei * 0.85))
    joystick_radius = max(55, min(110, int(min(gwid, ghei) * 0.075)))
    data = {
        "monitor": {"top": top, "left": left, "width": width, "height": height},
        "game_rect": {"x": gx, "y": gy, "width": gwid, "height": ghei},
        "joystick_center": [joystick_center[0], joystick_center[1]],
        "joystick_radius": joystick_radius,
        "color_ranges": {},
        "templates": {},
    }
    Path(config.CALIBRATION_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[calibrate] Window: left={left} top={top} width={width} height={height}")
    print(f"[calibrate] Game canvas: x={gx} y={gy} width={gwid} height={ghei}")
    print(f"[calibrate] Joystick center: {joystick_center}, radius={joystick_radius}")
    return data


def click_start_button(win: Any) -> None:
    """Click the red play/start button in the lower middle of the game menu."""

    try:
        if click_template_button(win, ("menu_play_button.png", "play_button.png"), threshold=0.70):
            time.sleep(2.0)
            return
        if click_red_play_button(win):
            time.sleep(2.0)
            return
        frame = _grab_window_bgr(win)
        gx, gy, gwid, ghei = detect_game_rect(frame)
        x = int(win.left + gx + gwid * 0.50)
        y = int(win.top + gy + ghei * 0.72)
        print(f"[start] Clicking estimated play button at ({x}, {y})")
        pyautogui.click(x, y)
        time.sleep(2.0)
    except Exception as exc:
        print(f"[start] Could not click play button: {exc}")


def _grab_window_bgr(win: Any) -> np.ndarray:
    screenshot = pyautogui.screenshot(region=(int(win.left), int(win.top), int(win.width), int(win.height)))
    rgb = np.array(screenshot)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_game_rect(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Detect the active portrait game canvas inside a possibly wide window."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    active = gray > 25
    col_counts = active.sum(axis=0)
    row_counts = active.sum(axis=1)
    cols = np.where(col_counts > frame.shape[0] * 0.18)[0]
    rows = np.where(row_counts > frame.shape[1] * 0.08)[0]
    if len(cols) < 100 or len(rows) < 100:
        return 0, 0, frame.shape[1], frame.shape[0]
    left, right = int(cols[0]), int(cols[-1] + 1)
    top, bottom = int(rows[0]), int(rows[-1] + 1)
    return left, top, right - left, bottom - top


def _crop_to_game(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rect = detect_game_rect(frame)
    x, y, w, h = rect
    return frame[y : y + h, x : x + w], rect


def click_template_button(win: Any, template_names: tuple[str, ...], threshold: float = 0.72) -> bool:
    """Click the best matching UI button template inside the game window."""

    try:
        frame_full = _grab_window_bgr(win)
        frame, (off_x, off_y, _, _) = _crop_to_game(frame_full)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best: tuple[float, int, int, int, int, str] | None = None
        for name in template_names:
            path = Path(config.TEMPLATE_DIR) / name
            if not path.exists():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            if image.ndim == 3 and image.shape[2] == 4:
                tmpl_bgr = image[:, :, :3]
                mask = cv2.threshold(image[:, :, 3], 5, 255, cv2.THRESH_BINARY)[1]
            else:
                tmpl_bgr = image[:, :, :3] if image.ndim == 3 else image
                mask = None
            tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY) if tmpl_bgr.ndim == 3 else tmpl_bgr
            for scale in (0.75, 0.9, 1.0, 1.1, 1.25):
                w = max(8, int(tmpl_gray.shape[1] * scale))
                h = max(8, int(tmpl_gray.shape[0] * scale))
                if w >= gray.shape[1] or h >= gray.shape[0]:
                    continue
                resized = cv2.resize(tmpl_gray, (w, h), interpolation=cv2.INTER_AREA)
                resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) if mask is not None else None
                try:
                    result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED, mask=resized_mask)
                except cv2.error:
                    result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if best is None or max_val > best[0]:
                    best = (float(max_val), max_loc[0], max_loc[1], w, h, name)
        if best and best[0] >= threshold:
            score, x, y, w, h, name = best
            click_x = int(win.left + off_x + x + w / 2)
            click_y = int(win.top + off_y + y + h / 2)
            print(f"[button] Clicking {name} score={score:.2f} at ({click_x}, {click_y})")
            pyautogui.click(click_x, click_y)
            return True
    except Exception as exc:
        print(f"[button] Template click failed: {exc}")
    return False


def click_red_play_button(win: Any) -> bool:
    """Find and click a large red circular play/retry button."""

    try:
        frame_full = _grab_window_bgr(win)
        frame, (off_x, off_y, _, _) = _crop_to_game(frame_full)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 90, 120), (12, 255, 255))
        red2 = cv2.inRange(hsv, (170, 90, 120), (180, 255, 255))
        red = cv2.bitwise_or(red1, red2)
        area_mask = np.zeros_like(red)
        area_mask[int(frame.shape[0] * 0.25) : int(frame.shape[0] * 0.90), int(frame.shape[1] * 0.20) : int(frame.shape[1] * 0.80)] = 255
        red = cv2.bitwise_and(red, area_mask)
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1200:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / max(h, 1)
            if 0.65 <= ratio <= 1.35 and 35 <= w <= 220 and 35 <= h <= 220:
                candidates.append((area, x + w / 2.0, y + h / 2.0))
        if not candidates:
            return False
        _, x, y = max(candidates, key=lambda item: item[0])
        click_x = int(win.left + off_x + x)
        click_y = int(win.top + off_y + y)
        print(f"[button] Clicking red play button at ({click_x}, {click_y})")
        pyautogui.click(click_x, click_y)
        return True
    except Exception as exc:
        print(f"[button] Red play click failed: {exc}")
    return False


def _template_folder_empty() -> bool:
    folder = Path(config.TEMPLATE_DIR)
    return not folder.exists() or not any(folder.glob("*.png"))


def _fallback_detect_player(frame: np.ndarray) -> Optional[tuple[float, float]]:
    """Estimate player position by finding the bright plane shape."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 160), (180, 80, 255))
    # Avoid UI text/icons at the very top and the joystick area at the bottom.
    mask = np.zeros_like(white)
    mask[int(frame.shape[0] * 0.16) : int(frame.shape[0] * 0.80), int(frame.shape[1] * 0.06) : int(frame.shape[1] * 0.94)] = 255
    white = cv2.bitwise_and(white, mask)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 45 <= area <= 1800:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / max(h, 1)
            if 6 <= w <= 75 and 10 <= h <= 95 and 0.20 <= ratio <= 2.20:
                cx, cy = x + w / 2.0, y + h / 2.0
                lower_bias = 1.0 - abs((cy / frame.shape[0]) - 0.55)
                center_bias = 1.0 - abs((cx / frame.shape[1]) - 0.50)
                score = area + 220.0 * lower_bias + 120.0 * center_bias
                candidates.append((score, cx, cy))
    if not candidates:
        return None
    _, x, y = max(candidates, key=lambda item: item[0])
    return x, y


def _fallback_detect_missiles(frame: np.ndarray) -> list[tuple[float, float]]:
    """Estimate missiles by finding small dark moving-object shapes."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (0, 20, 15), (180, 255, 105))
    # Ignore brown UI text, joystick, and window edges; gameplay threats are tiny dark shapes.
    mask = np.zeros_like(dark)
    top_limit = int(frame.shape[0] * 0.82)
    mask[:top_limit, int(frame.shape[1] * 0.08) : int(frame.shape[1] * 0.92)] = 255
    dark = cv2.bitwise_and(dark, mask)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    missiles: list[tuple[float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 8 <= area <= 350:
            x, y, w, h = cv2.boundingRect(contour)
            if 3 <= w <= 35 and 3 <= h <= 45:
                missiles.append((x + w / 2.0, y + h / 2.0))
    return missiles[:12]


def _fallback_detect_pickups(frame: np.ndarray) -> list[Detection]:
    """Find likely collectible/power-up circles by color."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_area = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_area[int(frame.shape[0] * 0.10) : int(frame.shape[0] * 0.84), int(frame.shape[1] * 0.06) : int(frame.shape[1] * 0.94)] = 255
    ranges = [
        ("star", (15, 80, 120), (45, 255, 255)),
        ("boost", (0, 100, 120), (15, 255, 255)),
        ("shield", (80, 50, 100), (120, 255, 255)),
    ]
    detections: list[Detection] = []
    for cls, low, high in ranges:
        mask = cv2.inRange(hsv, low, high)
        mask = cv2.bitwise_and(mask, mask_area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 60 <= area <= 4000:
                x, y, w, h = cv2.boundingRect(contour)
                ratio = w / max(h, 1)
                if 0.55 <= ratio <= 1.8 and 8 <= w <= 90 and 8 <= h <= 90:
                    detections.append(Detection(cls, x + w / 2.0, y + h / 2.0, 0.35, w, h))
    return detections


def _offset_detections(detections: list[Detection], off_x: int, off_y: int) -> list[Detection]:
    """Convert game-canvas local detections into captured-window coordinates."""

    return [
        Detection(det.cls, det.x + off_x, det.y + off_y, det.confidence, det.w, det.h)
        for det in detections
    ]


def run_ai(max_seconds: Optional[float] = None) -> None:
    """Run capture, detection, prediction, planning, and joystick control."""

    from capture import FrameCapture
    from controller import PCController
    from danger_map import DangerMap
    from detector import CalibratedDetector, Detection
    from planner import Planner
    from predictor import MissilePredictor
    from tracker import Tracker

    capture = FrameCapture()
    detector = CalibratedDetector()
    tracker = Tracker()
    predictor = MissilePredictor()
    danger_map = DangerMap(capture.monitor["width"], capture.monitor["height"])
    planner = Planner(danger_map)
    controller = PCController()
    started_at = time.perf_counter()
    frame_index = 0
    frame_delay = 1.0 / config.TARGET_FPS
    no_templates = _template_folder_empty()
    win = find_game_window()
    print("[ai] Pilot running. Press Ctrl+C to stop.")
    if no_templates:
        print("[ai] No PNG templates found; using limited fallback player detection.")

    try:
        while True:
            if emergency_stop_pressed():
                print("[ai] F8 emergency stop pressed.")
                break
            if max_seconds is not None and time.perf_counter() - started_at >= max_seconds:
                print(f"[ai] Test duration reached: {max_seconds:.1f}s")
                break
            loop_start = time.perf_counter()
            if not is_game_foreground():
                controller.release()
                if frame_index % 30 == 0:
                    print(f"[ai] Real game is not foreground, current foreground: {_foreground_title()!r}")
                    if win is not None:
                        focus_window(win)
                time.sleep(0.10)
                frame_index += 1
                continue

            frame = capture.grab()
            game_frame, (off_x, off_y, _, _) = _crop_to_game(frame)
            detections: list[Detection] = []
            if win is not None and frame_index % 60 == 0 and click_red_play_button(win):
                tracker = Tracker()
                controller.release()
                time.sleep(1.0)
                frame_index += 1
                continue

            fallback_player = _fallback_detect_player(game_frame)
            if fallback_player:
                detections.append(
                    Detection("player", fallback_player[0] + off_x, fallback_player[1] + off_y, 0.5, 40, 40)
                )
            for mx, my in _fallback_detect_missiles(game_frame):
                detections.append(Detection("missile", mx + off_x, my + off_y, 0.4, 18, 18))
            detections.extend(_offset_detections(_fallback_detect_pickups(game_frame), off_x, off_y))
            tracks = tracker.update(detections)
            player = tracker._get_player_track()
            if player is not None:
                elapsed = time.perf_counter() - started_at
                trajectories = predictor.predict_all(tracks, player.x, player.y, elapsed)
                danger_grid = danger_map.compute(trajectories, (player.x, player.y))
                dx, dy = planner.plan((player.x, player.y), tracks, danger_grid)
                controller.send(dx, dy)
                if frame_index % 10 == 0:
                    missiles = sum(1 for t in tracks if t.cls == "missile")
                    danger = danger_map.potential_at(danger_grid, player.x, player.y)
                    print(
                        f"[ai] frame={frame_index} player=({player.x:.0f},{player.y:.0f}) "
                        f"missiles={missiles} danger={danger:.2f} input=({dx:.2f},{dy:.2f}) "
                        f"capture={capture.capture_ms:.1f}ms"
                    )
            else:
                controller.release()
                if frame_index % 30 == 0:
                    print("[ai] Waiting for player detection...")
                if win is not None and frame_index % 90 == 0:
                    clicked = click_template_button(
                        win,
                        ("restart_button.png", "play_button.png", "menu_play_button.png"),
                        threshold=0.70,
                    )
                    if not clicked:
                        click_red_play_button(win)

            frame_index += 1
            sleep_for = frame_delay - (time.perf_counter() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[ai] Stop requested.")
    finally:
        controller.release()
        print("[ai] Joystick released.")


def main() -> None:
    """Find or launch the game, calibrate it, start gameplay, and run AI."""

    parser = argparse.ArgumentParser(description="Launch and play Missiles! automatically.")
    parser.add_argument("--duration", type=float, default=None, help="Optional run time in seconds for testing.")
    args = parser.parse_args()

    print("=" * 60)
    print("MISSILES! AI PILOT")
    print("=" * 60)
    win = find_game_window()
    if win is None:
        launch_game()
        win = wait_for_game_window(timeout=60.0)
    if win is None:
        shell = find_shell_window()
        if shell is not None:
            print(f"[error] Google Play Games is open, but the real Missiles window did not appear: {shell.title}")
            print("[error] Open Missiles from Google Play Games, then run this script again.")
        else:
            print("[error] Could not find Missiles! window. Open the game and run again.")
        sys.exit(1)
    if not focus_window(win):
        print("[error] Game window is not foreground. Click the game once and run again.")
        sys.exit(1)
    calibrate_from_window(win)
    click_start_button(win)
    print("[main] Starting AI in 3 seconds...")
    for value in (3, 2, 1):
        print(f"  {value}")
        time.sleep(1)
    run_ai(max_seconds=args.duration)


if __name__ == "__main__":
    main()
