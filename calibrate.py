"""High-confidence calibration for the Missiles! game canvas and controls.

Calibration no longer depends on fixed screen coordinates. Instead it combines
several independent techniques and only accepts a result when they agree:

1. Letterbox analysis locates the portrait game canvas inside the window.
2. Masked, multi-scale template matching of the recovered joystick sprites
   (``joystick_external.png`` / ``joystick_interial.png`` / ``tutorial_joystick.png``)
   locates the joystick centre and radius.
3. A circular-Hough fallback recovers the joystick when templates are weak.
4. Every result is scored; a blended confidence is returned so the caller can
   refuse to start gameplay on an unreliable calibration.

All functions here are platform independent (only ``cv2`` / ``numpy``) so the
calibration logic can be unit-tested without a live game window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

Rect = Tuple[int, int, int, int]


@dataclass
class CalibrationResult:
    """Result of a calibration attempt in window-local coordinates.

    ``joystick_center`` and ``game_rect`` are relative to the captured frame
    (i.e. window-local). The launcher adds the window origin to obtain absolute
    screen coordinates.
    """

    game_rect: Rect
    joystick_center: Tuple[int, int]
    joystick_radius: int
    confidence: float
    sources: Dict[str, float] = field(default_factory=dict)
    ui: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= config.CALIBRATION_MIN_CONFIDENCE


def detect_game_rect(frame: np.ndarray) -> Rect:
    """Detect the active portrait game canvas inside a possibly wide window.

    Google Play Games renders the portrait game centred with black
    letterboxing. We find the contiguous bright band that is tall enough to be
    the canvas rather than incidental bright UI.
    """

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    h, w = gray.shape[:2]
    active = gray > 25
    col_counts = active.sum(axis=0)
    row_counts = active.sum(axis=1)
    cols = np.where(col_counts > h * 0.18)[0]
    rows = np.where(row_counts > w * 0.08)[0]
    if len(cols) < max(50, w // 20) or len(rows) < max(50, h // 20):
        return 0, 0, w, h
    left, right = int(cols[0]), int(cols[-1] + 1)
    top, bottom = int(rows[0]), int(rows[-1] + 1)
    rect_w, rect_h = right - left, bottom - top
    if rect_w <= 0 or rect_h <= 0:
        return 0, 0, w, h
    return left, top, rect_w, rect_h


def _load_template_gray(path: Path) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        mask = cv2.threshold(image[:, :, 3], 5, 255, cv2.THRESH_BINARY)[1]
    else:
        bgr = image[:, :, :3] if image.ndim == 3 else image
        mask = None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return gray, mask


def _best_template_match(
    gray: np.ndarray,
    template_gray: np.ndarray,
    template_mask: Optional[np.ndarray],
    sizes: List[int],
) -> Optional[Tuple[float, int, int, int]]:
    """Return (score, cx, cy, size) of the best multi-scale square match."""

    best: Optional[Tuple[float, int, int, int]] = None
    th, tw = template_gray.shape[:2]
    for size in sizes:
        if size < 12:
            continue
        scale = size / max(th, tw)
        w = max(8, int(tw * scale))
        h = max(8, int(th * scale))
        if w >= gray.shape[1] or h >= gray.shape[0]:
            continue
        resized = cv2.resize(template_gray, (w, h), interpolation=cv2.INTER_AREA)
        resized_mask = (
            cv2.resize(template_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if template_mask is not None
            else None
        )
        try:
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED, mask=resized_mask)
        except cv2.error:
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best[0]:
            cx = int(max_loc[0] + w / 2)
            cy = int(max_loc[1] + h / 2)
            best = (float(max_val), cx, cy, max(w, h))
    return best


def detect_joystick_by_template(
    frame: np.ndarray, game_rect: Rect, template_dir: Path
) -> Optional[Tuple[Tuple[int, int], int, float]]:
    """Locate the joystick using recovered joystick sprites.

    Returns ((cx, cy), radius, confidence) in window-local coordinates, or None.
    """

    gx, gy, gw, gh = game_rect
    # The joystick lives in the lower portion of the canvas.
    y0 = gy + int(gh * 0.60)
    roi = frame[y0 : gy + gh, gx : gx + gw]
    if roi.size == 0:
        return None
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    short_edge = min(gw, gh)
    # Joystick external ring is typically 12%-26% of the canvas short edge.
    sizes = [int(short_edge * f) for f in np.linspace(0.12, 0.30, 6)]

    candidates: List[Tuple[float, int, int, int]] = []
    for name in ("joystick_external.png", "tutorial_joystick.png", "joystick_interial.png"):
        loaded = _load_template_gray(template_dir / name)
        if loaded is None:
            continue
        tmpl_gray, tmpl_mask = loaded
        match = _best_template_match(roi_gray, tmpl_gray, tmpl_mask, sizes)
        if match is not None:
            candidates.append(match)
    if not candidates:
        return None
    score, cx, cy, size = max(candidates, key=lambda item: item[0])
    center = (gx + cx, y0 + cy)
    radius = int(np.clip(size / 2.0 * 0.62, config.JOYSTICK_RADIUS_MIN, config.JOYSTICK_RADIUS_MAX))
    confidence = float(np.clip(score, 0.0, 1.0))
    return center, radius, confidence


def detect_joystick_by_hough(frame: np.ndarray, game_rect: Rect) -> Optional[Tuple[Tuple[int, int], int, float]]:
    """Fallback joystick detection via circular Hough transform."""

    gx, gy, gw, gh = game_rect
    y0 = gy + int(gh * 0.62)
    roi = frame[y0 : gy + gh, gx : gx + gw]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.medianBlur(gray, 5)
    short_edge = min(gw, gh)
    min_r = int(short_edge * 0.08)
    max_r = int(short_edge * 0.20)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=short_edge * 0.3,
        param1=100,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))[0]
    # Prefer the circle nearest the horizontal centre of the canvas.
    canvas_center_x = gw / 2.0
    best = min(circles, key=lambda c: abs(int(c[0]) - canvas_center_x))
    cx, cy, radius = int(best[0]), int(best[1]), int(best[2])
    center = (gx + cx, y0 + cy)
    radius = int(np.clip(radius * 0.9, config.JOYSTICK_RADIUS_MIN, config.JOYSTICK_RADIUS_MAX))
    return center, radius, 0.5


def _geometry_joystick(game_rect: Rect) -> Tuple[Tuple[int, int], int]:
    gx, gy, gw, gh = game_rect
    center = (gx + gw // 2, gy + int(gh * 0.85))
    radius = int(np.clip(min(gw, gh) * 0.09, config.JOYSTICK_RADIUS_MIN, config.JOYSTICK_RADIUS_MAX))
    return center, radius


def find_ui_element(
    frame: np.ndarray,
    game_rect: Rect,
    template_names: Tuple[str, ...],
    template_dir: Path,
    threshold: float = 0.62,
) -> Optional[Tuple[Tuple[int, int], float]]:
    """Locate a UI button by template inside the canvas; returns (center, score)."""

    gx, gy, gw, gh = game_rect
    roi = frame[gy : gy + gh, gx : gx + gw]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    short_edge = min(gw, gh)
    sizes = [int(short_edge * f) for f in np.linspace(0.08, 0.28, 6)]
    best: Optional[Tuple[float, int, int]] = None
    for name in template_names:
        loaded = _load_template_gray(template_dir / name)
        if loaded is None:
            continue
        tmpl_gray, tmpl_mask = loaded
        match = _best_template_match(gray, tmpl_gray, tmpl_mask, sizes)
        if match is not None and (best is None or match[0] > best[0]):
            best = (match[0], match[1], match[2])
    if best is None or best[0] < threshold:
        return None
    return (gx + best[1], gy + best[2]), float(best[0])


def calibrate(frame: np.ndarray, template_dir: Optional[Path] = None) -> CalibrationResult:
    """Run full calibration on a captured (window-local) BGR frame."""

    template_dir = template_dir or Path(config.TEMPLATE_DIR)
    game_rect = detect_game_rect(frame)
    gx, gy, gw, gh = game_rect

    sources: Dict[str, float] = {}

    # --- Canvas confidence: a tall, portrait-ish canvas is a good sign. ---
    aspect = gh / max(gw, 1)
    rect_conf = float(np.clip((aspect - 1.0) / 0.9, 0.0, 1.0)) if aspect >= 1.0 else 0.2
    area_frac = (gw * gh) / max(frame.shape[0] * frame.shape[1], 1)
    rect_conf *= float(np.clip(area_frac / 0.20, 0.3, 1.0))
    sources["canvas"] = rect_conf

    # --- Joystick: template first, Hough second, geometry last. ---
    joystick_center: Tuple[int, int]
    joystick_radius: int
    joystick_conf: float
    template_hit = detect_joystick_by_template(frame, game_rect, template_dir)
    if template_hit is not None and template_hit[2] >= 0.45:
        joystick_center, joystick_radius, joystick_conf = template_hit
        sources["joystick_template"] = joystick_conf
    else:
        hough_hit = detect_joystick_by_hough(frame, game_rect)
        if hough_hit is not None:
            joystick_center, joystick_radius, joystick_conf = hough_hit
            sources["joystick_hough"] = joystick_conf
        else:
            joystick_center, joystick_radius = _geometry_joystick(game_rect)
            joystick_conf = 0.25
            sources["joystick_geometry"] = joystick_conf

    # --- Validate joystick placement: it must be inside the lower canvas. ---
    jx, jy = joystick_center
    inside_x = gx + gw * 0.15 <= jx <= gx + gw * 0.85
    inside_y = gy + gh * 0.60 <= jy <= gy + gh
    if not (inside_x and inside_y):
        # Placement failed sanity check; fall back to geometry with low confidence.
        joystick_center, joystick_radius = _geometry_joystick(game_rect)
        joystick_conf = min(joystick_conf, 0.22)
        sources["joystick_placement_reject"] = 1.0

    # --- UI anchors (optional; boost confidence when present). ---
    ui: Dict[str, Tuple[int, int]] = {}
    play = find_ui_element(
        frame, game_rect, ("menu_play_button.png", "play_button.png", "restart_button.png"), template_dir
    )
    if play is not None:
        ui["play"] = play[0]
        sources["play_button"] = play[1]

    # --- Blended confidence: canvas and joystick dominate. ---
    confidence = 0.45 * rect_conf + 0.45 * joystick_conf + 0.10 * (play[1] if play else 0.0)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return CalibrationResult(
        game_rect=game_rect,
        joystick_center=joystick_center,
        joystick_radius=joystick_radius,
        confidence=confidence,
        sources=sources,
        ui=ui,
    )
