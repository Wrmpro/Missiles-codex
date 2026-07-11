"""Template based object detector for Missiles! sprites."""

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
            "star": [],
            "shield": [],
            "boost": [],
        }
        self._load_templates(Path(template_dir))

    def _class_from_name(self, name: str) -> Optional[str]:
        lowered = name.lower()
        if re.fullmatch(r"plane\d+\.png", lowered):
            return "player"
        if re.fullmatch(r"missile\d+\.png", lowered):
            return "missile"
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

    def _detect_by_template(
        self, gray: np.ndarray, obj_type: str, threshold: Optional[float] = None
    ) -> List[Detection]:
        detections: List[Detection] = []
        class_thresholds = {
            "player": max(self.threshold, 0.75),
            "missile": 0.58,
            "star": 0.62,
            "shield": 0.62,
            "boost": 0.62,
        }
        threshold = threshold if threshold is not None else class_thresholds.get(obj_type, self.threshold)
        for tmpl, mask, tw, th in self.templates.get(obj_type, []):
            for scale in (0.8, 1.0, 1.2):
                w, h = max(4, int(tw * scale)), max(4, int(th * scale))
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
                ys, xs = np.where(result >= threshold)
                for x, y in zip(xs, ys):
                    detections.append(
                        Detection(obj_type, x + w / 2.0, y + h / 2.0, float(result[y, x]), w, h)
                    )
        return detections

    def _nms(self, detections: List[Detection], min_distance: float = 25.0) -> List[Detection]:
        kept: List[Detection] = []
        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if all(
                det.cls != other.cls
                or np.hypot(det.x - other.x, det.y - other.y) >= min_distance
                for other in kept
            ):
                kept.append(det)
        return kept

    def detect(self, frame: np.ndarray, include_classes: Optional[Iterable[str]] = None) -> List[Detection]:
        """Detect all configured object classes in a BGR frame."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections: List[Detection] = []
        classes = list(include_classes) if include_classes is not None else list(self.templates)
        for obj_type in classes:
            detections.extend(self._detect_by_template(gray, obj_type))
        return self._nms(detections)

    # In your detector.py or tracker.py, add this validation:

def _get_player_track(self):
    players = [t for t in self.tracks.values() 
               if t.cls == 'player' and t.is_alive]
    if not players:
        return None
    
    # Filter out false positives (player should be in bottom 60% of screen)
    valid_players = [p for p in players 
                     if p.y > self.game_height * 0.4]  # Must be below 40% of screen
    
    if not valid_players:
        return None
    
    return max(valid_players, key=lambda t: t.confidence)