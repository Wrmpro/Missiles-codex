"""Missile trajectory simulation matching the game's steering-limited homing."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import config
from tracker import Track


def _delta_angle(current_deg: float, target_deg: float) -> float:
    """Return Unity-style shortest signed angle difference in [-180, 180]."""

    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


class MissilePredictor:
    """Predict missile paths from current track state."""

    def get_speed_multiplier(self, elapsed_seconds: float) -> float:
        """Return the game's speed multiplier for the elapsed run time."""

        return 1.0 + 0.05 * int(elapsed_seconds // 30.0)

    def _estimate_initial_state(self, track: Track) -> Tuple[float, float]:
        heading = track.heading
        omega = 0.0
        if len(track.history) >= 3:
            p0, p1, p2 = track.history[-3:]
            h1 = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            h2 = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            heading = h2
            omega = _delta_angle(h1, h2) / max(config.PREDICTION_DT, 1e-6)
        return heading, omega

    def simulate_single(
        self,
        mx: float,
        my: float,
        heading_deg: float,
        omega_deg: float,
        player_x: float,
        player_y: float,
        speed_mult: float,
    ) -> List[Tuple[float, float]]:
        """Simulate one missile trajectory for the configured horizon."""

        x, y = mx, my
        heading = heading_deg
        omega = omega_deg
        points: List[Tuple[float, float]] = []
        for _ in range(config.PREDICTION_STEPS):
            target_angle = math.degrees(math.atan2(player_y - y, player_x - x))
            angle_diff = _delta_angle(heading, target_angle)
            torque = angle_diff * config.MISSILE_ANGULAR_ACCELERATION
            drag = omega * config.MISSILE_ANGULAR_DRAG
            omega += (torque - drag) * config.PREDICTION_DT
            omega = max(
                -config.MISSILE_ANGULAR_MAX_SPEED,
                min(config.MISSILE_ANGULAR_MAX_SPEED, omega),
            )
            heading += omega * config.PREDICTION_DT
            radians = math.radians(heading)
            x += math.cos(radians) * config.MISSILE_BASE_SPEED * speed_mult * config.PREDICTION_DT
            y += math.sin(radians) * config.MISSILE_BASE_SPEED * speed_mult * config.PREDICTION_DT
            points.append((x, y))
        return points

    def predict_all(
        self, tracks: List[Track], player_x: float, player_y: float, elapsed: float
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Predict trajectories for all active missile tracks."""

        speed_mult = self.get_speed_multiplier(elapsed)
        trajectories: Dict[int, List[Tuple[float, float]]] = {}
        for track in tracks:
            if track.cls != "missile":
                continue
            heading, omega = self._estimate_initial_state(track)
            trajectories[track.id] = self.simulate_single(
                track.x, track.y, heading, omega, player_x, player_y, speed_mult
            )
        return trajectories
