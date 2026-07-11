"""Missile trajectory simulation matching the game's steering-limited homing."""

from __future__ import annotations

from collections import deque
import math
from typing import Dict, List, Tuple

import numpy as np

import config
from tracker import Track


def _delta_angle(current_deg: float, target_deg: float) -> float:
    """Return Unity-style shortest signed angle difference in [-180, 180]."""

    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


class PhysicsEstimator:
    """Estimate missile steering and speed from recent observed tracks."""

    def __init__(self, window: int = 60) -> None:
        self.base_speed = float(config.MISSILE_BASE_SPEED)
        self.angular_max_speed = float(config.MISSILE_ANGULAR_MAX_SPEED)
        self.angular_acceleration = float(config.MISSILE_ANGULAR_ACCELERATION)
        self.confidence = 0.0
        self._speed_history: deque[float] = deque(maxlen=window)
        self._omega_history: deque[float] = deque(maxlen=window)
        self._alpha_history: deque[float] = deque(maxlen=window)

    @staticmethod
    def _ema(previous: float, observed: float, alpha: float) -> float:
        return previous + alpha * (observed - previous)

    @staticmethod
    def _percentile(values: deque[float], p: float) -> float:
        sorted_values = sorted(values)
        if not sorted_values:
            return 0.0
        index = int(np.clip(round((len(sorted_values) - 1) * p), 0, len(sorted_values) - 1))
        return float(sorted_values[index])

    def update_from_tracks(self, tracks: List[Track], dt: float, speed_mult: float) -> None:
        """Update estimates from observed missile states; preserve conservative fallback."""

        if dt <= 1e-6:
            return

        missiles = [track for track in tracks if track.cls == "missile"]
        if not missiles:
            self.confidence = max(0.0, self.confidence * 0.95)
            return

        for track in missiles:
            speed = math.hypot(track.vx, track.vy) / dt
            if math.isfinite(speed) and speed > 1.0:
                self._speed_history.append(speed)

            if len(track.history) >= 3:
                p0, p1, p2 = track.history[-3:]
                h1 = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                h2 = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                omega = abs(_delta_angle(h1, h2)) / dt
                if math.isfinite(omega) and omega > 1e-3:
                    self._omega_history.append(omega)

            if len(track.history) >= 4:
                p0, p1, p2, p3 = track.history[-4:]
                h0 = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                h1 = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                h2 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
                omega0 = _delta_angle(h0, h1) / dt
                omega1 = _delta_angle(h1, h2) / dt
                alpha = abs(omega1 - omega0) / dt
                if math.isfinite(alpha) and alpha > 1e-3:
                    self._alpha_history.append(alpha)

        speed_count = len(self._speed_history)
        if speed_count < 6:
            self.confidence = max(0.0, self.confidence * 0.98)
            return

        speed_values = list(self._speed_history)
        mean_speed = float(sum(speed_values) / speed_count)
        variance = float(sum((value - mean_speed) ** 2 for value in speed_values) / speed_count)
        speed_cv = math.sqrt(max(variance, 0.0)) / max(mean_speed, 1.0)
        count_confidence = min(1.0, speed_count / 20.0)
        stability_confidence = max(0.0, 1.0 - min(1.0, speed_cv))
        self.confidence = count_confidence * stability_confidence

        normalized_speed = mean_speed / max(speed_mult, 0.5)
        observed_base_speed = float(np.clip(normalized_speed, 80.0, 650.0))
        blend = 0.03 + 0.32 * self.confidence
        self.base_speed = self._ema(self.base_speed, observed_base_speed, blend)

        if self._omega_history:
            # Use upper quantile of observed turn-rate to stay conservative under power-up spikes.
            observed_omega = self._percentile(self._omega_history, 0.85)
            observed_omega = float(np.clip(observed_omega, 60.0, 360.0))
            self.angular_max_speed = self._ema(
                self.angular_max_speed, observed_omega, 0.02 + 0.20 * self.confidence
            )

        if self._alpha_history:
            # Use upper quantile of observed angular acceleration for bounded steering simulation.
            observed_alpha = self._percentile(self._alpha_history, 0.85)
            observed_alpha = float(np.clip(observed_alpha, 8.0, 90.0))
            self.angular_acceleration = self._ema(
                self.angular_acceleration, observed_alpha, 0.02 + 0.20 * self.confidence
            )

    def parameters(self) -> Tuple[float, float, float]:
        """Return confidence-blended parameters with safe fallback to config defaults."""

        conf = float(np.clip(self.confidence, 0.0, 1.0))
        base_speed = (1.0 - conf) * config.MISSILE_BASE_SPEED + conf * self.base_speed
        angular_max = (1.0 - conf) * config.MISSILE_ANGULAR_MAX_SPEED + conf * self.angular_max_speed
        angular_acc = (1.0 - conf) * config.MISSILE_ANGULAR_ACCELERATION + conf * self.angular_acceleration
        return float(base_speed), float(angular_max), float(angular_acc)


class MissilePredictor:
    """Predict missile paths from current track state."""

    def __init__(self) -> None:
        self._estimator = PhysicsEstimator()

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
        base_speed: float,
        angular_max_speed: float,
        angular_acceleration: float,
    ) -> List[Tuple[float, float]]:
        """Simulate one missile trajectory for the configured horizon."""

        x, y = mx, my
        heading = heading_deg
        omega = omega_deg
        points: List[Tuple[float, float]] = []
        for _ in range(config.PREDICTION_STEPS):
            target_angle = math.degrees(math.atan2(player_y - y, player_x - x))
            angle_diff = _delta_angle(heading, target_angle)
            torque = angle_diff * angular_acceleration
            drag = omega * config.MISSILE_ANGULAR_DRAG
            omega += (torque - drag) * config.PREDICTION_DT
            omega = max(
                -angular_max_speed,
                min(angular_max_speed, omega),
            )
            heading += omega * config.PREDICTION_DT
            radians = math.radians(heading)
            x += math.cos(radians) * base_speed * speed_mult * config.PREDICTION_DT
            y += math.sin(radians) * base_speed * speed_mult * config.PREDICTION_DT
            points.append((x, y))
        return points

    def predict_all(
        self, tracks: List[Track], player_x: float, player_y: float, elapsed: float
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Predict trajectories for all active missile tracks."""

        speed_mult = self.get_speed_multiplier(elapsed)
        self._estimator.update_from_tracks(tracks, config.PREDICTION_DT, speed_mult)
        base_speed, angular_max, angular_acc = self._estimator.parameters()
        trajectories: Dict[int, List[Tuple[float, float]]] = {}
        for track in tracks:
            if track.cls != "missile":
                continue
            heading, omega = self._estimate_initial_state(track)
            trajectories[track.id] = self.simulate_single(
                track.x,
                track.y,
                heading,
                omega,
                player_x,
                player_y,
                speed_mult,
                base_speed,
                angular_max,
                angular_acc,
            )
        return trajectories
