"""Shared configuration for the Missiles! AI pilot.

All tunables live here so the runtime behaviour can be adjusted without touching
module logic. Values are grouped by the subsystem that consumes them.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Missile physics (used as safe fallbacks; PhysicsEstimator refines these online)
# ---------------------------------------------------------------------------
MISSILE_ANGULAR_MAX_SPEED = 180.0
MISSILE_ANGULAR_ACCELERATION = 25.0
MISSILE_ANGULAR_DRAG = 2.5
MISSILE_BASE_SPEED = 200.0
PLAYER_SPEED = 300.0

# ---------------------------------------------------------------------------
# Detection / template matching
# ---------------------------------------------------------------------------
TEMPLATE_MATCH_THRESHOLD = 0.65
# Extracted Unity sprites are stored at native (large) resolution but render
# small in-game. These fractions describe the plausible on-screen size of each
# object as a fraction of the game canvas' shorter edge. The detector uses them
# to build a physically-plausible scale pyramid instead of matching at 1.0x.
OBJECT_SIZE_FRACTIONS = {
    "player": (0.06, 0.16),
    "missile": (0.02, 0.075),
    "star": (0.04, 0.11),
    "shield": (0.05, 0.14),
    "boost": (0.04, 0.12),
    "cloud": (0.20, 0.95),
}
DETECTION_SCALE_STEPS = 4
# Region of interest inside the game canvas where gameplay objects can appear,
# expressed as (top, bottom, left, right) fractions. Excludes HUD and joystick.
GAMEPLAY_ROI = (0.10, 0.86, 0.03, 0.97)
# Maximum objects kept per class per frame (guards against false-positive floods).
MAX_DETECTIONS = {"player": 1, "missile": 16, "star": 8, "shield": 4, "boost": 4}

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
TRACK_MAX_AGE = 12
# Player is expected in the lower part of the canvas; reject "player" tracks above this.
PLAYER_MIN_Y_FRACTION = 0.30

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
PREDICTION_DT = 0.033
PREDICTION_HORIZON = 2.0
PREDICTION_STEPS = int(PREDICTION_HORIZON / PREDICTION_DT)

# ---------------------------------------------------------------------------
# Danger map (artificial potential field)
# ---------------------------------------------------------------------------
APF_ETA = 1000.0
APF_D0 = 200.0
DANGER_CRITICAL = 0.8
DANGER_MODERATE = 0.4
GRID_RESOLUTION = 15

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
# Radius (px) inside which a pickup collect is only attempted when survival is high.
COLLECT_SAFE_DANGER = 0.30
# Fraction of canvas short-edge used as the "safe recentering" comfort band.
EDGE_MARGIN_FRACTION = 0.12

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
# Exponential smoothing applied to the joystick vector to remove jitter while
# still allowing fast turns. 1.0 = no smoothing.
CONTROLLER_SMOOTHING = 0.55
# Minimum magnitude before the joystick is considered "engaged".
CONTROLLER_DEADZONE = 0.04

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
TARGET_FPS = 30

# ---------------------------------------------------------------------------
# Game-state machine timings (seconds)
# ---------------------------------------------------------------------------
STATE_LAUNCH_TIMEOUT = 60.0
STATE_START_TIMEOUT = 8.0
STATE_GAMEOVER_CONFIRM_FRAMES = 4      # consecutive detections before accepting game over
STATE_PLAYING_CONFIRM_FRAMES = 3       # consecutive player detections before accepting play
STATE_RESTART_COOLDOWN = 1.2           # min seconds between restart button clicks
STATE_RECALIBRATE_INTERVAL = 90.0      # periodic calibration re-verification while playing

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
# Minimum confidence required from calibration before gameplay may start.
CALIBRATION_MIN_CONFIDENCE = 0.45
# Joystick radius clamps (px), as safety around the template-derived value.
JOYSTICK_RADIUS_MIN = 40
JOYSTICK_RADIUS_MAX = 130

# ---------------------------------------------------------------------------
# Defaults / paths (overridden by calibration.json at runtime)
# ---------------------------------------------------------------------------
MONITOR_BOUNDS = {"top": 0, "left": 0, "width": 1280, "height": 720}
JOYSTICK_CENTER = (640, 612)
JOYSTICK_MAX_RADIUS = 80
CALIBRATION_FILE = str(BASE_DIR / "calibration.json")
TEMPLATE_DIR = str(BASE_DIR / "assets" / "templates")
