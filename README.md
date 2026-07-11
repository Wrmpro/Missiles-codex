# Missiles! AI Pilot — Single Technical Document

Autonomous controller for the Google Play Games version of **Missiles!**, designed as a deterministic, production-style bot that preserves stability while improving long-term survival behavior.

---

## 1) What this project is

This bot captures the game window, detects game entities, tracks motion over time, predicts missile trajectories, computes a danger field, plans a safe movement direction, and sends joystick-style mouse input.

The architecture is modular and intentionally conservative:

- `capture.py`: window/screen capture
- `detector.py`: template-based object detection
- `tracker.py`: track association and velocity/history estimation
- `predictor.py`: steering-limited missile trajectory simulation
- `danger_map.py`: APF-based danger map generation
- `planner.py`: deterministic direction selection balancing survival and objectives
- `controller.py`: mouse joystick control
- `auto_calibrate.py`: launch, calibration, orchestration, runtime loop
- `config.py`: all shared tuning constants

---

## 2) Setup and run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put template PNG assets into:

```text
/home/runner/work/Missiles-codex/Missiles-codex/assets/templates/
```

Use names such as `plane1.png`, `missile1.png`, `star1.png`, `shield1.png`, `boost1.png`.

3. Run the bot:

```bash
python auto_calibrate.py
```

Press `Ctrl+C` to stop safely.

---

## 3) Runtime pipeline (frame-by-frame)

1. Capture current frame (`capture.py`)
2. Detect objects (`detector.py`)
3. Update tracks and per-object history (`tracker.py`)
4. Predict missile futures (`predictor.py`)
5. Build danger field from predicted trajectories (`danger_map.py`)
6. Choose deterministic movement vector (`planner.py`)
7. Send joystick command (`controller.py`)
8. Apply restart/recovery logic (`auto_calibrate.py`)

---

## 4) Deterministic planning and survival logic

### 4.1 Missile prediction (`predictor.py`)

Missiles are simulated using steering-limited dynamics:

- heading update from angular velocity
- angular velocity update from torque-drag model
- bounded angular speed
- forward motion from missile speed and heading

Recent upgrades keep this deterministic while adapting online:

- confidence-gated `PhysicsEstimator` learns observed missile speed/turn behavior from track history
- conservative fallback to `config.py` defaults when confidence is low
- no random sampling

### 4.2 Danger field (`danger_map.py`)

Danger map remains APF-based and now includes:

- temporal weighting (near-future danger weighted more strongly)
- immediate collision-zone pressure
- short persistence blending with previous frame danger to reduce oscillation

### 4.3 Planner (`planner.py`)

Planner now evaluates deterministic candidate directions using:

- danger at short look-ahead horizons
- boundary proximity penalty
- train-preservation reward (prefer keeping missiles behind)
- front exposure penalty (avoid missiles in forward path)
- momentum penalty (reduce jitter and unnecessary reversals)

Candidate directions are generated with a deterministic golden-angle pattern.

---

## 5) Does this code survive longer?

**Likely yes in many scenarios, but this must be validated empirically.**

Reason: the recent logic explicitly improves long-horizon stability (adaptive missile physics, temporal danger weighting, anti-jitter planning, train-preservation scoring), which generally supports longer survival.

However, survival time depends on:

- template quality and detection accuracy
- game speed/powerup conditions
- system FPS and input latency
- calibration quality

So the correct claim is: **improved survival strategy is implemented; measured survival gain requires benchmark runs.**

---

## 6) How to benchmark survival-time improvements

Use fixed conditions for before/after comparisons:

1. Same machine and display settings
2. Same templates
3. Same calibration
4. Same runtime duration budget per trial
5. At least 20+ runs per version

Track:

- mean survival time
- median survival time
- p90 survival time
- missile count over time (train stability proxy)

Accept enhancement only if survival metrics improve without instability regressions.

---

## 7) Recreate this implementation from this single document

A model can recreate the current behavior by implementing these exact constraints:

1. Keep module boundaries and public interfaces unchanged.
2. Keep deterministic planning (no randomness).
3. Use steering-limited missile simulation with bounded angular velocity and drag.
4. Add confidence-based online estimation of missile speed/turn parameters from recent track history.
5. Blend learned parameters with conservative defaults when confidence is low.
6. Keep APF core formula and add deterministic temporal weighting by prediction step.
7. Add immediate collision-zone amplification and danger persistence smoothing.
8. Score planner candidates using multi-horizon danger, edge margin, collectible utility, train-preservation reward, and momentum/front-risk penalties.
9. Preserve safe fallbacks: if prediction/planning confidence drops, revert to conservative behavior.
10. Do not alter controller API or joystick contract; only change target direction generation.

---

## 8) Operational notes

- Window detection intentionally ignores common desktop apps.
- Missing templates reduce detection quality and survival.
- Bot reliability is higher when FPS is stable near target.
- For documentation-only changes, no runtime logic changes are required.
