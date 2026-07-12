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
    missed_frames: int = 0
    hits: int = 1
    ax: float = 0.0
    ay: float = 0.0
    stability: float = 0.0
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def predict(self) -> None:
        """Advance by one frame using a constant velocity model."""

        self.x += self.vx
        self.y += self.vy
        self.age += 1
        self.missed_frames += 1
        self.confidence *= 0.97
        self.history.append((self.x, self.y))
        self.history = self.history[-20:]

    def update(self, det: Detection) -> None:
        """Update this track with an assigned detection."""

        alpha = 0.4
        new_vx = det.x - self.x
        new_vy = det.y - self.y
        prev_vx, prev_vy = self.vx, self.vy
        self.vx = alpha * new_vx + (1.0 - alpha) * self.vx
        self.vy = alpha * new_vy + (1.0 - alpha) * self.vy
        self.ax = self.vx - prev_vx
        self.ay = self.vy - prev_vy
        if abs(self.vx) + abs(self.vy) > 0.01:
            self.heading = math.degrees(math.atan2(self.vy, self.vx))
        self.x = det.x
        self.y = det.y
        self.confidence = det.confidence
        self.age = 0
        self.missed_frames = 0
        self.hits += 1
        consistency = 1.0 / (1.0 + math.hypot(self.ax, self.ay))
        persistence = min(1.0, self.hits / 8.0)
        self.stability = 0.65 * persistence + 0.35 * consistency
        self.history.append((self.x, self.y))
        self.history = self.history[-20:]

    @property
    def is_alive(self) -> bool:
        """Return True while the track is still fresh enough to use."""

        return self.missed_frames < config.TRACK_MAX_AGE


class Tracker:
    """Associate detections with existing tracks using Hungarian matching."""

    def __init__(self, game_height: Optional[int] = None) -> None:
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        # Canvas height (px) in the same coordinate space as detections. Used to
        # reject "player" tracks that appear implausibly high on screen.
        self.game_height: Optional[int] = game_height

    def configure(self, game_height: int) -> None:
        """Set the canvas height so player validation can use a real bound."""

        if game_height and game_height > 0:
            self.game_height = int(game_height)

    def _is_valid_player(self, track: Track) -> bool:
        """A real player sits in the lower part of the canvas and is fresh."""

        if not track.is_alive or track.cls != "player":
            return False
        if self.game_height:
            min_y = self.game_height * config.PLAYER_MIN_Y_FRACTION
            if track.y < min_y:
                return False
        return True

    def _get_player_track(self) -> Optional[Track]:
        players = [t for t in self.tracks.values() if self._is_valid_player(t)]
        if not players and self.game_height is None:
            # Only before the canvas height is known do we accept any live
            # player track; once known, spatial validation is authoritative so
            # false positives high on screen are rejected.
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

    @staticmethod
    def _gate(track: Track, det: Detection) -> float:
        """Distance gate for association, scaled by class and observed speed."""

        base = 70.0 if det.cls == "missile" else 120.0
        dynamic = min(80.0, track.speed * 1.8)
        return base + dynamic

    def _is_stable_missile(self, track: Track) -> bool:
        if not track.is_alive:
            return False
        if track.cls != "missile":
            return False
        if track.hits < 3:
            return False
        if track.confidence < 0.53:
            return False
        if track.speed < 0.8:
            return False
        if track.speed > 120.0:
            return False
        return track.stability >= 0.28

    def stable_tracks(self) -> List[Track]:
        """Return tracks suitable for planning and prediction."""

        stable: List[Track] = []
        for track in self.tracks.values():
            if not track.is_alive:
                continue
            if track.cls == "missile":
                if self._is_stable_missile(track):
                    stable.append(track)
            else:
                stable.append(track)
        return stable

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
                        distance = float(np.hypot(track.x - det.x, track.y - det.y))
                        if distance <= self._gate(track, det):
                            costs[i, j] = distance
            rows, cols = linear_sum_assignment(costs)
            matched_dets = set()
            for row, col in zip(rows, cols):
                if costs[row, col] < 1e5:
                    track_list[row].update(detections[col])
                    matched_dets.add(col)
            for idx, det in enumerate(detections):
                if idx not in matched_dets:
                    self._new_track(det)

        self.tracks = {tid: t for tid, t in self.tracks.items() if t.is_alive}
        return list(self.tracks.values())
