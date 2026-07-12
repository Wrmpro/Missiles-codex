"""Decide joystick direction from danger, motion strategy and collectibles.

The original planner scored a ring of candidate directions purely on danger and
tended to fly straight until it died, or oscillate. This version keeps that
robust danger-scoring core but layers intelligent, *continuous* motion on top:

* a persistent orbit/spiral field around the canvas centre so the plane is
  always moving (circular when comfortable, expanding when too central,
  contracting when near an edge -> "safe recentering");
* escape-corridor evaluation (multi-step danger integral along each candidate);
* collision-train management (reward keeping missiles trailing so they group
  and collide instead of surrounding the plane);
* survival-gated collecting (only divert to stars/shields/boosts when the
  expected survival probability stays high);
* momentum/continuity so the joystick vector adapts smoothly instead of
  jittering or going constant.

``plan`` keeps its original signature for drop-in compatibility.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

import config
from danger_map import DangerMap
from tracker import Track


class Planner:
    """Blend survival, continuous motion, collision-shaping and collecting."""

    def __init__(self, danger_map: DangerMap) -> None:
        self.danger_map = danger_map
        golden_angle = math.radians(137.507764)
        self._candidate_dirs = [
            (float(math.cos(index * golden_angle)), float(math.sin(index * golden_angle)))
            for index in range(48)
        ]
        self._candidate_dirs.append((0.0, 0.0))
        self._last_direction = (0.0, 0.0)
        self._orbit_sign = 1.0
        self._orbit_flip_cooldown = 0
        self._current_grid: Optional[np.ndarray] = None
        short_edge = min(self.danger_map.width, self.danger_map.height)
        self._radius_low = 0.26 * short_edge
        self._radius_high = 0.44 * short_edge
        self._edge_margin = config.EDGE_MARGIN_FRACTION * short_edge

    # ------------------------------------------------------------------
    # Continuous-motion field (circular / spiral / orbit / recenter)
    # ------------------------------------------------------------------
    def _motion_field(self, px: float, py: float, danger: float) -> Tuple[float, float]:
        """Return a unit vector describing the preferred perpetual-motion heading."""

        cx = self.danger_map.width / 2.0
        cy = self.danger_map.height / 2.0
        rx, ry = px - cx, py - cy
        dist_r = math.hypot(rx, ry)
        if dist_r < 1e-3:
            rx, ry, dist_r = 1.0, 0.0, 1.0
        r_hat = (rx / dist_r, ry / dist_r)

        # Tangential (curl) component -> circular / orbiting motion.
        tangential = (-self._orbit_sign * r_hat[1], self._orbit_sign * r_hat[0])

        # Radial component -> spiral out when too central, spiral in near edge.
        radial = 0.0
        if dist_r > self._radius_high:
            radial = -1.0  # contract toward centre (safe recentering)
        elif dist_r < self._radius_low:
            radial = 0.55   # expand outward to open room

        # Under high danger, tighten the spiral (favour contraction) to buy space.
        if danger > config.DANGER_MODERATE:
            radial = min(radial, -0.2)

        mx = tangential[0] + radial * r_hat[0]
        my = tangential[1] + radial * r_hat[1]
        norm = math.hypot(mx, my)
        if norm < 1e-6:
            return tangential
        return mx / norm, my / norm

    def _maybe_flip_orbit(self, px: float, py: float, danger_grid: np.ndarray) -> None:
        """Flip orbit direction only when the current curl leads into danger."""

        if self._orbit_flip_cooldown > 0:
            self._orbit_flip_cooldown -= 1
            return
        motion = self._motion_field(px, py, self.danger_map.potential_at(danger_grid, px, py))
        look = config.PLAYER_SPEED * 0.7
        fwd = self.danger_map.potential_at(
            danger_grid,
            float(np.clip(px + motion[0] * look, 0, self.danger_map.width - 1)),
            float(np.clip(py + motion[1] * look, 0, self.danger_map.height - 1)),
        )
        if fwd > config.DANGER_CRITICAL:
            self._orbit_sign *= -1.0
            self._orbit_flip_cooldown = 12

    # ------------------------------------------------------------------
    # Escape corridors: integrate danger along a candidate direction
    # ------------------------------------------------------------------
    def _corridor_danger(self, px: float, py: float, dx: float, dy: float) -> Tuple[float, float]:
        """Return (weighted danger integral, min boundary margin) along a ray."""

        grid = self._current_grid
        total = 0.0
        weight_sum = 0.0
        min_margin = 1e9
        for i, frac in enumerate((0.30, 0.60, 0.90, 1.25)):
            step = config.PLAYER_SPEED * frac
            x = float(np.clip(px + dx * step, 0, self.danger_map.width - 1))
            y = float(np.clip(py + dy * step, 0, self.danger_map.height - 1))
            w = 1.0 / (i + 1)
            total += w * self.danger_map.potential_at(grid, x, y)
            weight_sum += w
            margin = min(x, y, self.danger_map.width - x, self.danger_map.height - y)
            min_margin = min(min_margin, margin)
        return total / max(weight_sum, 1e-6), min_margin

    # ------------------------------------------------------------------
    # Collision-train shaping
    # ------------------------------------------------------------------
    @staticmethod
    def _train_score(px: float, py: float, dx: float, dy: float, tracks: List[Track]) -> Tuple[float, float]:
        """Reward keeping missiles trailing (train) and penalise front exposure."""

        trailing_reward = 0.0
        front_penalty = 0.0
        for track in tracks:
            if track.cls != "missile":
                continue
            rel_x = track.x - px
            rel_y = track.y - py
            dist = math.hypot(rel_x, rel_y)
            if dist < 1e-6:
                continue
            to_missile_x = rel_x / dist
            to_missile_y = rel_y / dist
            aheadness = dx * to_missile_x + dy * to_missile_y

            speed = math.hypot(track.vx, track.vy)
            closing = 0.5
            if speed > 1e-6:
                heading_x = track.vx / speed
                heading_y = track.vy / speed
                closing = max(0.0, heading_x * (-to_missile_x) + heading_y * (-to_missile_y))

            if aheadness > 0.35 and dist < 170.0:
                front_penalty += ((170.0 - dist) / 170.0) * (0.6 + 0.4 * closing)
            if aheadness < -0.15:
                trailing_reward += (-aheadness) * min(1.0, 220.0 / dist) * closing
        return trailing_reward, front_penalty

    # ------------------------------------------------------------------
    # Survival-gated target selection
    # ------------------------------------------------------------------
    def _select_target(
        self,
        tracks: List[Track],
        danger_grid: np.ndarray,
        player_pos: Tuple[float, float],
        has_shield: bool,
        has_boost: bool,
    ) -> Optional[Tuple[float, float]]:
        px, py = player_pos
        priorities = []
        if not has_shield:
            priorities.append("shield")
        if not has_boost:
            priorities.append("boost")
        priorities.append("star")
        for cls in priorities:
            candidates = [
                t
                for t in tracks
                if t.cls == cls
                and self.danger_map.potential_at(danger_grid, t.x, t.y) < config.COLLECT_SAFE_DANGER
            ]
            if candidates:
                closest = min(candidates, key=lambda t: math.hypot(t.x - px, t.y - py))
                return closest.x, closest.y
        return None

    def _momentum_penalty(self, dx: float, dy: float, danger: float) -> float:
        last_x, last_y = self._last_direction
        if abs(last_x) + abs(last_y) < 1e-6 or abs(dx) + abs(dy) < 1e-6:
            return 0.0
        cos_turn = float(np.clip(dx * last_x + dy * last_y, -1.0, 1.0))
        # Smooth turns whenever possible; permit aggressive turns under danger.
        return (1.0 - cos_turn) * max(0.0, 1.0 - danger) * 0.45

    # ------------------------------------------------------------------
    # Main planning entry point
    # ------------------------------------------------------------------
    def plan(
        self,
        player_pos: Tuple[float, float],
        tracks: List[Track],
        danger_grid: np.ndarray,
        has_shield: bool = False,
        has_boost: bool = False,
    ) -> Tuple[float, float]:
        """Return normalized joystick direction as dx, dy (continuous motion)."""

        self._current_grid = danger_grid
        px, py = player_pos
        danger = self.danger_map.potential_at(danger_grid, px, py)
        self._maybe_flip_orbit(px, py, danger_grid)
        motion = self._motion_field(px, py, danger)
        target = self._select_target(tracks, danger_grid, player_pos, has_shield, has_boost)

        best_score = -1e9
        best_dir = motion
        for dx, dy in self._candidate_dirs:
            corridor, min_margin = self._corridor_danger(px, py, dx, dy)
            d_here = self.danger_map.potential_at(
                danger_grid,
                float(np.clip(px + dx * config.PLAYER_SPEED * 0.45, 0, self.danger_map.width - 1)),
                float(np.clip(py + dy * config.PLAYER_SPEED * 0.45, 0, self.danger_map.height - 1)),
            )
            boundary_penalty = max(0.0, (self._edge_margin - min_margin) / max(self._edge_margin, 1.0))
            trailing_reward, front_penalty = self._train_score(px, py, dx, dy, tracks)
            momentum_penalty = self._momentum_penalty(dx, dy, danger)

            # Continuous-motion reward: align with the orbit/spiral field, and
            # discourage standing still so the plane keeps circulating.
            motion_align = dx * motion[0] + dy * motion[1]
            stillness_penalty = 0.6 if (abs(dx) + abs(dy)) < 1e-6 else 0.0

            target_score = 0.0
            if target is not None and danger < config.DANGER_CRITICAL:
                before = math.hypot(target[0] - px, target[1] - py)
                after = math.hypot(
                    target[0] - float(np.clip(px + dx * config.PLAYER_SPEED * 0.9, 0, self.danger_map.width - 1)),
                    target[1] - float(np.clip(py + dy * config.PLAYER_SPEED * 0.9, 0, self.danger_map.height - 1)),
                )
                target_score = float(np.clip((before - after) / 180.0, -1.0, 1.0))

            score = -(2.4 * corridor + 1.0 * d_here + 0.9 * boundary_penalty
                      + 1.4 * front_penalty + momentum_penalty + stillness_penalty)
            score += 0.7 * trailing_reward

            # Motion / collecting weights depend on how threatened we are.
            if danger > config.DANGER_CRITICAL:
                score += 0.10 * motion_align            # survival dominates
            elif danger > config.DANGER_MODERATE:
                score += 0.35 * motion_align + 0.25 * target_score
            else:
                score += 0.55 * motion_align + 0.9 * target_score

            if score > best_score:
                best_score = score
                best_dir = (dx, dy)

        # Never emit a zero vector while alive: keep circulating.
        if abs(best_dir[0]) + abs(best_dir[1]) < 1e-6:
            best_dir = motion

        smoothed = self._smooth(best_dir)
        self._last_direction = smoothed
        return smoothed

    def _smooth(self, direction: Tuple[float, float]) -> Tuple[float, float]:
        """Blend toward the new direction for jitter-free but responsive turns."""

        last_x, last_y = self._last_direction
        alpha = config.CONTROLLER_SMOOTHING
        nx = alpha * direction[0] + (1.0 - alpha) * last_x
        ny = alpha * direction[1] + (1.0 - alpha) * last_y
        norm = math.hypot(nx, ny)
        if norm < 1e-6:
            return direction
        return nx / norm, ny / norm
