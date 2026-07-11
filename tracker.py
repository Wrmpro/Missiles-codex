"""Object tracking for detections across frames."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from detector import Detection


@dataclass
class Track:
    """Tracked object state with smoothed velocity and history."""

    id: int
    cls: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    confidence: float = 0.0
    age: int = 0
    hits: int = 1
    history: List[Tuple[float, float]] = field(default_factory=list)
    

    def predict(self) -> None:
        """Advance by one frame using a constant velocity model."""

        self.x += self.vx
        self.y += self.vy
        self.age += 1
        self.history.append((self.x, self.y))
        self.history = self.history[-20:]

    # In tracker.py, add to Track class:
    def is_valid_player(self):
        """Check if this track is a valid player track."""
        if len(self.history) < 3:
            return False
    
    # Player should be in bottom half of screen
        if self.y < 400:  # Adjust based on your screen height
            return False
    
    # Player speed should be reasonable (not stationary like a cloud)
        if self.speed < 10:  # Too slow = probably not player
            return False
    
        return True
    
    def update(self, det: Detection) -> None:
        """Update this track with an assigned detection."""

        alpha = 0.4
        new_vx = det.x - self.x
        new_vy = det.y - self.y
        self.vx = alpha * new_vx + (1.0 - alpha) * self.vx
        self.vy = alpha * new_vy + (1.0 - alpha) * self.vy
        if abs(self.vx) + abs(self.vy) > 0.01:
            self.heading = math.degrees(math.atan2(self.vy, self.vx))
        self.x = det.x
        self.y = det.y
        self.confidence = det.confidence
        self.age = 0
        self.hits += 1
        self.history.append((self.x, self.y))
        self.history = self.history[-20:]

    @property
    def is_alive(self) -> bool:
        """Return True while the track is still fresh enough to use."""

        return self.age < config.TRACK_MAX_AGE


class Tracker:
    """Associate detections with existing tracks using Hungarian matching."""

    def __init__(self) -> None:
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _get_player_track(self) -> Optional[Track]:
        players = [t for t in self.tracks.values() if t.cls == "player" and t.is_alive]
        if not players:
            return None
        return max(players, key=lambda t: (t.confidence, t.hits))

    def _new_track(self, det: Detection) -> None:
        self.tracks[self._next_id] = Track(
            id=self._next_id,
            cls=det.cls,
            x=det.x,
            y=det.y,
            confidence=det.confidence,
            history=[(det.x, det.y)],
        )
        self._next_id += 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks and return active tracks."""

        for track in list(self.tracks.values()):
            track.predict()

        track_list = list(self.tracks.values())
        if not track_list:
            for det in detections:
                self._new_track(det)
            return list(self.tracks.values())

        if detections:
            costs = np.full((len(track_list), len(detections)), 1e6, dtype=np.float32)
            for i, track in enumerate(track_list):
                for j, det in enumerate(detections):
                    if track.cls == det.cls:
                        costs[i, j] = np.hypot(track.x - det.x, track.y - det.y)
            rows, cols = linear_sum_assignment(costs)
            matched_tracks = set()
            matched_dets = set()
            for row, col in zip(rows, cols):
                if costs[row, col] <= 120.0:
                    track_list[row].update(detections[col])
                    matched_tracks.add(row)
                    matched_dets.add(col)
            for idx, det in enumerate(detections):
                if idx not in matched_dets:
                    self._new_track(det)

        self.tracks = {tid: t for tid, t in self.tracks.items() if t.is_alive}
        return list(self.tracks.values())
