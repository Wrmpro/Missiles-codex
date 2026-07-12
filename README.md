# Missiles-codex Handoff (Single Rebuild Document)

This repository is an autonomous pilot for the Google Play Games version of **Missiles!** on Windows.
This document is the complete handoff for rebuilding, running, and understanding the current codebase.

## 1) What this project does

At runtime the bot:
1. Finds or launches the game window.
2. Focuses it and calibrates game canvas / joystick / UI with confidence scoring.
3. Captures frames and crops to the game canvas.
4. Detects player, missiles, and pickups (recovered sprite templates first, color/shape fallback second).
5. Tracks objects across frames with spatial validation.
6. Predicts curved missile trajectories with online physics estimation.
7. Builds a danger map.
8. Plans continuous, adaptive movement (circular/spiral/orbit, escape corridors, collision trains, safe recentering).
9. Drives the virtual joystick with the mouse.

All launch/play/death/restart logic is coordinated by an explicit finite-state
machine (`game_state.py`), so the bot recovers reliably from lost focus, stale
coordinates, and death screens, and restarts after every death.

Entrypoint: `/home/runner/work/Missiles-codex/Missiles-codex/auto_calibrate.py`

## 2) Environment requirements

- OS: **Windows** (required by `pywin32`, window APIs, and desktop mouse control flow).
- Python: 3.10+ recommended.
- Google Play Games installed with **Missiles!** available.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Dependencies are declared in:
`/home/runner/work/Missiles-codex/Missiles-codex/requirements.txt`

## 3) Recreate the repository layout

Core files:
- `auto_calibrate.py` - launcher + FSM-driven runtime loop
- `game_state.py` - finite-state machine (launch/calibrate/menu/play/game-over)
- `calibrate.py` - template-based canvas/joystick/UI calibration with confidence
- `config.py` - shared constants and paths
- `capture.py` - monitor frame grabbing
- `detector.py` - template-based object detection (canvas-aware scaling, ROI, cloud negatives)
- `tracker.py` - detection-to-track association with player validation
- `predictor.py` - missile trajectory simulation + online physics estimation
- `danger_map.py` - potential-field danger grid
- `planner.py` - continuous adaptive joystick direction planning
- `controller.py` - joystick mouse control (smoothed, deadzoned)
- `tests_offline.py` - platform-independent smoke/logic tests (no game required)
- `calibration.json` - persisted monitor/joystick calibration

Assets:
- `/home/runner/work/Missiles-codex/Missiles-codex/assets/templates/` - PNG templates used for UI/object matching.

## 4) Build and run workflow

### Step A - Validate Python syntax

```powershell
python -m compileall -q .
```

### Step A2 - Run offline logic tests (no game/Windows required)

```powershell
python tests_offline.py
```

This exercises calibration, detection, tracking, prediction, danger map,
planner and the state machine on synthetic data. Requires `numpy`, `opencv`,
`scipy` (a headless subset of `requirements.txt`).

### Step B - Launch bot

```powershell
python auto_calibrate.py
```

Optional timed run for testing:

```powershell
python auto_calibrate.py --duration 60
```

### Step C - Runtime behavior

The script will:
- Locate or launch Missiles!.
- Focus the correct game window.
- Write/update `calibration.json`.
- Attempt to click a play/restart button.
- Start the AI loop.

Stop methods:
- `Ctrl+C` in terminal.
- `F8` emergency stop.

The joystick is released in shutdown paths.

## 5) Detection assets and naming

Templates are loaded from:
`/home/runner/work/Missiles-codex/Missiles-codex/assets/templates`

Current detector class mapping in `detector.py`:
- `plane<digits>.png` -> player
- `missile<digits>.png` -> missile
- `star.png` -> star
- `shield_power_up.png` or `shield 1.png` -> shield
- `boost_power_up.png` -> boost

If templates are unavailable, fallback color/shape detection is used (lower accuracy).

## 6) High-level data flow (module handoff)

1. `GameStateMachine.update(Observation)` decides the action for the tick.
2. `FrameCapture.grab()` returns a BGR frame; the loop crops to the game canvas.
3. `CalibratedDetector.detect()` runs sprite-template detection; color/shape
   fallback fills in classes with no template hit.
4. `Tracker.update()` maintains object tracks with velocity/history and rejects
   implausible (e.g. high-on-screen) player detections.
5. `MissilePredictor.predict_all()` simulates curved future missile positions,
   refining physics online.
6. `DangerMap.compute()` converts trajectories into normalized danger.
7. `Planner.plan()` chooses a continuous movement vector (orbit/spiral/escape).
8. `PCController.send(dx, dy)` applies smoothed movement through the mouse.

## 7) Important runtime configuration

Edit `/home/runner/work/Missiles-codex/Missiles-codex/config.py` to tune:
- Prediction horizon and FPS
- Missile physics defaults
- Danger thresholds
- Grid resolution
- Default monitor/joystick fallback values

## 8) Troubleshooting

- Bot cannot find game window:
  - Open Missiles! manually in Google Play Games, then rerun.
- Bot sees launcher shell but not game:
  - Ensure actual gameplay window is open (not only shell/home).
- Poor detection:
  - Verify PNG templates exist in `assets/templates`.
- Wrong movement center/radius:
  - Re-run `python auto_calibrate.py` to refresh `calibration.json`.

## 9) Quick rebuild checklist

1. Clone repo.
2. Install Python dependencies from `requirements.txt`.
3. Ensure `assets/templates` contains expected PNGs.
4. Run `python -m compileall -q .`.
5. Run `python auto_calibrate.py`.
6. Verify `calibration.json` updates and AI loop starts.
