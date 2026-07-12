"""Template based object detector for Missiles! sprites.

The recovered Unity sprites are stored at their native (large) authoring
resolution, but they render *small* in the captured game window. Matching them
at ~1.0x scale (as the original code did) essentially never fires on real
gameplay frames. This detector instead builds a physically-plausible scale
pyramid per class from the game-canvas size, restricts the search to the
gameplay region of interest, uses cloud sprites as *negative* evidence so
clouds are never mistaken for missiles, and caps detections per class to keep
false positives from flooding the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

import config


@dataclass
class Detection:
    """A single detected game object in window-local coordinates."""

    cls: str
    x: float
    y: float
    confidence: float
    w: int
    h: int


Template = Tuple[np.ndarray, Optional[np.ndarray], int, int]


class CalibratedDetector:
    """Load sprite templates and detect game objects by masked template match."""

    def __init__(
        self,
        template_dir: str = config.TEMPLATE_DIR,
        threshold: float = config.TEMPLATE_MATCH_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.templates: Dict[str, List[Template]] = {
            "player": [],
            "missile": [],
            "cloud": [],
            "star": [],
            "shield": [],
            "boost": [],
        }
        # Canvas short edge in px; drives the plausible-size pyramid. Set via
        # ``configure`` once calibration is known. A sane default keeps the
        # detector usable before calibration.
        self._canvas_short = 500.0
        self._load_templates(Path(template_dir))

    def configure(self, canvas_w: int, canvas_h: int) -> None:
        """Tell the detector the game-canvas size so it can scale templates."""

        short = float(min(max(canvas_w, 1), max(canvas_h, 1)))
        if short > 20:
            self._canvas_short = short

    def _class_from_name(self, name: str) -> Optional[str]:
        lowered = name.lower()
        if re.fullmatch(r"plane\d+\.png", lowered):
            return "player"
        if re.fullmatch(r"missile\d+\.png", lowered):
            return "missile"
        if re.fullmatch(r"cloud\d+\.png", lowered) or re.fullmatch(r"menu_cloud\d+.*\.png", lowered):
            return "cloud"
        if lowered in {"star.png"}:
            return "star"
        if lowered in {"shield_power_up.png", "shield 1.png"}:
            return "shield"
        if lowered in {"boost_power_up.png"}:
            return "boost"
        return None

    def _load_templates(self, root: Path) -> None:
        if not root.exists():
            print(f"[detector] Template folder missing: {root}")
            return
        for path in sorted(root.glob("*.png")):
            obj_type = self._class_from_name(path.name)
            if not obj_type:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"[detector] Could not read template: {path}")
                continue
            if image.ndim == 3 and image.shape[2] == 4:
                alpha = image[:, :, 3]
                bgr = image[:, :, :3]
                mask = cv2.threshold(alpha, 5, 255, cv2.THRESH_BINARY)[1]
            else:
                bgr = image[:, :, :3] if image.ndim == 3 else image
                mask = None
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            h, w = gray.shape[:2]
            if w >= 6 and h >= 6:
                self.templates[obj_type].append((gray, mask, w, h))
        counts = ", ".join(f"{k}={len(v)}" for k, v in self.templates.items())
        print(f"[detector] Loaded templates: {counts}")

    def _target_long_edges(self, obj_type: str) -> List[int]:
        """Plausible on-screen long-edge sizes (px) for a class."""

        low_f, high_f = config.OBJECT_SIZE_FRACTIONS.get(obj_type, (0.04, 0.14))
        low = self._canvas_short * low_f
        high = self._canvas_short * high_f
        steps = max(2, config.DETECTION_SCALE_STEPS)
        return [int(round(v)) for v in np.linspace(low, high, steps)]

    @staticmethod
    def _extract_peaks(result: np.ndarray, threshold: float, min_distance: int = 6) -> List[Tuple[int, int, float]]:
        """Return local maxima above threshold to avoid duplicate raw matches."""

        if result.size == 0:
            return []
        kernel_size = max(3, 2 * min_distance + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        maxima = cv2.dilate(result, kernel)
        mask = (result >= threshold) & np.isclose(result, maxima, atol=1e-6)
        ys, xs = np.where(mask)
        peaks = [(int(x), int(y), float(result[y, x])) for x, y in zip(xs, ys)]
        peaks.sort(key=lambda item: item[2], reverse=True)
        return peaks[:60]

    def _detect_by_template(
        self, gray: np.ndarray, obj_type: str, threshold: Optional[float] = None
    ) -> List[Detection]:
        detections: List[Detection] = []
        class_thresholds = {
            "player": max(self.threshold, 0.72),
            "missile": 0.60,
            "cloud": 0.60,
            "star": 0.60,
            "shield": 0.60,
            "boost": 0.60,
        }
        threshold = threshold if threshold is not None else class_thresholds.get(obj_type, self.threshold)
        target_edges = self._target_long_edges(obj_type)
        for tmpl, mask, tw, th in self.templates.get(obj_type, []):
            long_edge = max(tw, th)
            for target in target_edges:
                if target < 6:
                    continue
                scale = target / float(long_edge)
                w, h = max(4, int(round(tw * scale))), max(4, int(round(th * scale)))
                if w >= gray.shape[1] or h >= gray.shape[0]:
                    continue
                resized = cv2.resize(tmpl, (w, h), interpolation=cv2.INTER_AREA)
                resized_mask = (
                    cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    if mask is not None
                    else None
                )
                try:
                    result = cv2.matchTemplate(
                        gray, resized, cv2.TM_CCOEFF_NORMED, mask=resized_mask
                    )
                except cv2.error:
                    result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                for x, y, score in self._extract_peaks(result, threshold):
                    detections.append(Detection(obj_type, x + w / 2.0, y + h / 2.0, score, w, h))
        return detections

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        ax1, ay1 = a.x - a.w / 2.0, a.y - a.h / 2.0
        ax2, ay2 = a.x + a.w / 2.0, a.y + a.h / 2.0
        bx1, by1 = b.x - b.w / 2.0, b.y - b.h / 2.0
        bx2, by2 = b.x + b.w / 2.0, b.y + b.h / 2.0
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = a.w * a.h + b.w * b.h - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    def _reject_cloud_overlaps(self, missiles: List[Detection], clouds: List[Detection]) -> List[Detection]:
        """Reject missile candidates that overlap known cloud detections."""

        kept: List[Detection] = []
        for missile in missiles:
            overlaps_cloud = False
            for cloud in clouds:
                if self._iou(missile, cloud) > 0.18:
                    overlaps_cloud = True
                    break
            if not overlaps_cloud:
                kept.append(missile)
        return kept

    def _nms(self, detections: List[Detection], iou_threshold: float = 0.35) -> List[Detection]:
        kept: List[Detection] = []
        by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.cls, []).append(det)
        for cls_dets in by_class.values():
            for det in sorted(cls_dets, key=lambda d: d.confidence, reverse=True):
                if all(self._iou(det, other) < iou_threshold for other in kept if other.cls == det.cls):
                    kept.append(det)
        return kept

    def _cap_per_class(self, detections: List[Detection]) -> List[Detection]:
        """Keep only the most confident N detections per class."""

        by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.cls, []).append(det)
        capped: List[Detection] = []
        for cls, items in by_class.items():
            limit = config.MAX_DETECTIONS.get(cls)
            items.sort(key=lambda d: d.confidence, reverse=True)
            capped.extend(items[:limit] if limit is not None else items)
        return capped

    @staticmethod
    def _roi_mask(shape: Tuple[int, int]) -> np.ndarray:
        """Build a mask for the gameplay region (excludes HUD and joystick)."""

        h, w = shape[:2]
        top, bottom, left, right = config.GAMEPLAY_ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * top) : int(h * bottom), int(w * left) : int(w * right)] = 255
        return mask

    def detect(
        self,
        frame: np.ndarray,
        include_classes: Optional[Iterable[str]] = None,
        use_roi: bool = True,
    ) -> List[Detection]:
        """Detect all configured object classes in a BGR game-canvas frame."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if use_roi:
            mask = self._roi_mask(gray.shape)
            gray = cv2.bitwise_and(gray, mask)

        requested = (
            list(include_classes)
            if include_classes is not None
            else [cls for cls in self.templates.keys() if cls != "cloud"]
        )
        classes = list(requested)
        if "missile" in requested and "cloud" not in classes and self.templates.get("cloud"):
            classes.append("cloud")

        raw: Dict[str, List[Detection]] = {}
        for obj_type in classes:
            raw[obj_type] = self._detect_by_template(gray, obj_type)

        missiles = raw.get("missile", [])
        clouds = raw.get("cloud", [])
        if missiles and clouds:
            raw["missile"] = self._reject_cloud_overlaps(missiles, clouds)

        merged = [det for cls, items in raw.items() for det in items if cls in requested]
        return self._cap_per_class(self._nms(merged))
