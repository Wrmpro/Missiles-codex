"""Offline smoke/logic tests for platform-independent modules.

These do NOT require Windows, a live game, pyautogui, or a display. They
exercise calibration, detection, tracking, prediction, danger map, planner and
the game-state machine on synthetic data to guard against regressions.

Run: python tests_offline.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config
import calibrate as calib
from detector import CalibratedDetector, Detection
from tracker import Tracker
from predictor import MissilePredictor
from danger_map import DangerMap
from planner import Planner
from game_state import Action, GameStateMachine, Observation, State

FAILURES: list[str] = []


def check(name: str, cond: bool) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        FAILURES.append(name)


def _synthetic_window(w: int = 900, h: int = 700, canvas_w: int = 400) -> np.ndarray:
    """A dark window with a bright centered portrait canvas (letterboxed)."""

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (w - canvas_w) // 2
    frame[:, x0 : x0 + canvas_w] = (60, 60, 60)
    return frame


def test_calibration_canvas() -> None:
    frame = _synthetic_window()
    x, y, cw, ch = calib.detect_game_rect(frame)
    check("canvas width ~400", abs(cw - 400) <= 20)
    check("canvas is full height", ch >= 680)
    result = calib.calibrate(frame, template_dir=Path(config.TEMPLATE_DIR))
    check("calibration returns a result", isinstance(result, calib.CalibrationResult))
    jx, jy = result.joystick_center
    check("joystick x inside canvas", x <= jx <= x + cw)
    check("joystick y in lower canvas", jy >= y + ch * 0.55)
    check(
        "joystick radius clamped",
        config.JOYSTICK_RADIUS_MIN <= result.joystick_radius <= config.JOYSTICK_RADIUS_MAX,
    )
    check("confidence in [0,1]", 0.0 <= result.confidence <= 1.0)


def test_detector_scaling() -> None:
    det = CalibratedDetector()
    det.configure(400, 700)
    edges = det._target_long_edges("missile")
    low, high = config.OBJECT_SIZE_FRACTIONS["missile"]
    check("missile target sizes scale with canvas", edges[0] <= edges[-1] and edges[-1] <= 400 * high + 2)
    # Detection on a blank frame should not hallucinate objects.
    blank = np.zeros((700, 400, 3), dtype=np.uint8)
    dets = det.detect(blank)
    check("no detections on blank frame", len(dets) == 0)


def test_detector_caps_and_nms() -> None:
    det = CalibratedDetector()
    raw = [Detection("missile", 100 + i * 0.5, 100, 0.9 - i * 0.01, 18, 18) for i in range(30)]
    capped = det._cap_per_class(det._nms(raw))
    check("nms collapses near-duplicates", len(capped) < 30)
    players = [Detection("player", 50 + i, 50, 0.8, 40, 40) for i in range(5)]
    capped_players = det._cap_per_class(players)
    check("player capped to 1", sum(1 for d in capped_players if d.cls == "player") == 1)


def test_tracker_player_validation() -> None:
    tracker = Tracker(game_height=700)
    # A "player" high on screen must be rejected by spatial validation.
    tracker.update([Detection("player", 200, 50, 0.9, 40, 40)])
    check("high player rejected", tracker._get_player_track() is None or tracker._get_player_track().y >= 210)
    tracker2 = Tracker(game_height=700)
    tracker2.update([Detection("player", 200, 600, 0.9, 40, 40)])
    check("low player accepted", tracker2._get_player_track() is not None)


def test_tracker_missile_association() -> None:
    tracker = Tracker(game_height=700)
    x = 100.0
    for _ in range(6):
        tracker.update([Detection("missile", x, 100, 0.8, 18, 18)])
        x += 8.0
    missiles = [t for t in tracker.tracks.values() if t.cls == "missile"]
    check("single missile track maintained", len(missiles) == 1)
    check("missile velocity positive", missiles[0].vx > 1.0)
    check("stable_tracks includes moving missile", any(t.cls == "missile" for t in tracker.stable_tracks()))


def test_predictor_curved() -> None:
    tracker = Tracker(game_height=700)
    # Missile moving right, player below-left -> homing should curve downward.
    for i in range(6):
        tracker.update([Detection("missile", 100 + i * 10, 100, 0.8, 18, 18)])
    predictor = MissilePredictor()
    tracks = tracker.stable_tracks()
    traj = predictor.predict_all(tracks, 120, 400, elapsed=5.0)
    check("prediction produced", len(traj) >= 1)
    for path in traj.values():
        check("trajectory has horizon steps", len(path) == config.PREDICTION_STEPS)
        ys = [p[1] for p in path]
        check("missile curves toward player (y increases)", ys[-1] > ys[0])
    check("physics confidence in [0,1]", 0.0 <= predictor.confidence <= 1.0)
    check("speed multiplier grows over time", predictor.get_speed_multiplier(60) > predictor.get_speed_multiplier(0))


def test_danger_map() -> None:
    dm = DangerMap(400, 700)
    traj = {1: [(200.0, 200.0)] * config.PREDICTION_STEPS}
    grid = dm.compute(traj, (200.0, 650.0))
    check("danger grid normalized", 0.0 <= float(grid.max()) <= 1.0 + 1e-6)
    near = dm.potential_at(grid, 200, 200)
    far = dm.potential_at(grid, 380, 650)
    check("danger higher near missile", near > far)
    gx, gy = dm.gradient_at(grid, 205, 205)
    check("gradient finite", np.isfinite(gx) and np.isfinite(gy))


def test_planner_continuous_and_safe() -> None:
    dm = DangerMap(400, 700)
    planner = Planner(dm)
    # No threats -> planner should still move (continuous motion, non-zero).
    grid = dm.compute({}, (200.0, 350.0))
    dx, dy = planner.plan((200.0, 350.0), [], grid)
    check("planner moves with no threats", (abs(dx) + abs(dy)) > 0.1)

    # A missile right on the player -> planner should steer away from it.
    tracker = Tracker(game_height=700)
    for i in range(5):
        tracker.update([Detection("missile", 210 - i, 355 - i, 0.8, 18, 18)])
    tracks = tracker.stable_tracks()
    traj = MissilePredictor().predict_all(tracks, 200, 350, 5.0)
    danger_grid = dm.compute(traj, (200.0, 350.0))
    ddx, ddy = planner.plan((200.0, 350.0), tracks, danger_grid)
    check("planner outputs unit-ish vector", 0.5 <= (ddx * ddx + ddy * ddy) ** 0.5 <= 1.01)


def test_planner_pickup_gating() -> None:
    dm = DangerMap(400, 700)
    planner = Planner(dm)
    grid = dm.compute({}, (200.0, 350.0))
    # Star far away, low danger -> should be selectable.
    from tracker import Track

    star = Track(id=99, cls="star", x=210.0, y=200.0, confidence=0.9)
    target = planner._select_target([star], grid, (200.0, 350.0), has_shield=False, has_boost=False)
    check("safe star is targeted", target is not None)


def test_state_machine_full_cycle() -> None:
    fsm = GameStateMachine()
    t = 0.0

    def step(**kw) -> Action:
        nonlocal t
        t += 0.1
        return fsm.update(Observation(now=t, **kw))

    # Not foreground -> focus.
    check("focus when not foreground", step(foreground=False) == Action.FOCUS_WINDOW)
    # Foreground -> calibrate.
    step(foreground=True, calibrated=False)
    check("calibrating state", fsm.state == State.CALIBRATING)
    check("asks to calibrate", step(foreground=True, calibrated=False) == Action.CALIBRATE)
    # Calibrated, no player -> menu.
    step(foreground=True, calibrated=True, player_present=False)
    check("menu after calibration", fsm.state == State.MENU)
    # In menu -> clicks play.
    act = step(foreground=True, calibrated=True, player_present=False, menu_signal=True)
    check("clicks play from menu", act == Action.CLICK_PLAY)
    check("starting state", fsm.state == State.STARTING)
    # Player appears repeatedly -> playing.
    last = Action.WAIT
    for _ in range(config.STATE_PLAYING_CONFIRM_FRAMES + 1):
        last = step(foreground=True, calibrated=True, player_present=True)
    check("runs AI when playing", last == Action.RUN_AI and fsm.state == State.PLAYING)
    # Death: gameover signal persists -> game over then restart.
    for _ in range(config.STATE_GAMEOVER_CONFIRM_FRAMES + 1):
        step(foreground=True, calibrated=True, player_present=False, gameover_signal=True)
    check("reaches game over", fsm.state in (State.GAME_OVER, State.STARTING))
    # Advance until a restart click is emitted.
    saw_restart = False
    for _ in range(40):
        act = step(foreground=True, calibrated=True, player_present=False, gameover_signal=True)
        if act == Action.CLICK_RESTART:
            saw_restart = True
            break
    check("restart is clicked after death", saw_restart)


def main() -> int:
    test_calibration_canvas()
    test_detector_scaling()
    test_detector_caps_and_nms()
    test_tracker_player_validation()
    test_tracker_missile_association()
    test_predictor_curved()
    test_danger_map()
    test_planner_continuous_and_safe()
    test_planner_pickup_gating()
    test_state_machine_full_cycle()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("All offline tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
