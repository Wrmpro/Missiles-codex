"""Decide joystick direction from danger and collectibles."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

import config
from danger_map import DangerMap
from tracker import Track


class Planner:
    """Blend survival force with attraction toward safe useful targets."""

    def __init__(self, danger_map: DangerMap) -> None:
        self.danger_map = danger_map
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        self._candidate_dirs = [(float(np.cos(a)), float(np.sin(a))) for a in angles]
        self._candidate_dirs.append((0.0, 0.0))

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
                if t.cls == cls and self.danger_map.potential_at(danger_grid, t.x, t.y) < config.DANGER_MODERATE
            ]
            if candidates:
                closest = min(candidates, key=lambda t: math.hypot(t.x - px, t.y - py))
                return closest.x, closest.y
        return None

    def plan(
        self,
        player_pos: Tuple[float, float],
        tracks: List[Track],
        danger_grid: np.ndarray,
        has_shield: bool = False,
        has_boost: bool = False,
    ) -> Tuple[float, float]:
        """Return normalized joystick direction as dx, dy."""

        px, py = player_pos
        danger = self.danger_map.potential_at(danger_grid, px, py)
        target = self._select_target(tracks, danger_grid, player_pos, has_shield, has_boost)

        best_score = -1e9
        best_dir = (0.0, 0.0)
        # Look ahead over plausible near-future plane positions. This is more stable than
        # reacting only to the current danger gradient.
        for dx, dy in self._candidate_dirs:
            pos1 = (
                np.clip(px + dx * config.PLAYER_SPEED * 0.45, 0, self.danger_map.width - 1),
                np.clip(py + dy * config.PLAYER_SPEED * 0.45, 0, self.danger_map.height - 1),
            )
            pos2 = (
                np.clip(px + dx * config.PLAYER_SPEED * 0.90, 0, self.danger_map.width - 1),
                np.clip(py + dy * config.PLAYER_SPEED * 0.90, 0, self.danger_map.height - 1),
            )
            d1 = self.danger_map.potential_at(danger_grid, pos1[0], pos1[1])
            d2 = self.danger_map.potential_at(danger_grid, pos2[0], pos2[1])
            margin = min(pos2[0], pos2[1], self.danger_map.width - pos2[0], self.danger_map.height - pos2[1])
            boundary_penalty = max(0.0, (90.0 - margin) / 90.0)
            target_score = 0.0
            if target is not None and danger < config.DANGER_CRITICAL:
                before = math.hypot(target[0] - px, target[1] - py)
                after = math.hypot(target[0] - pos2[0], target[1] - pos2[1])
                target_score = np.clip((before - after) / 180.0, -1.0, 1.0)
            score = -(2.2 * d1 + 1.4 * d2 + 0.8 * boundary_penalty)
            if danger < config.DANGER_MODERATE:
                score += 1.0 * target_score
            elif danger < config.DANGER_CRITICAL:
                score += 0.25 * target_score
            if score > best_score:
                best_score = score
                best_dir = (dx, dy)

        if best_dir != (0.0, 0.0):
            return best_dir

        grad_x, grad_y = self.danger_map.gradient_at(danger_grid, px, py)
        survival = np.array([-grad_x, -grad_y], dtype=np.float32)
        target_force = np.zeros(2, dtype=np.float32)
        if target is not None:
            target_force = np.array([target[0] - px, target[1] - py], dtype=np.float32)
            norm = np.linalg.norm(target_force)
            if norm > 1e-6:
                target_force /= norm

        survival_norm = np.linalg.norm(survival)
        if survival_norm > 1e-6:
            survival /= survival_norm

        if danger > config.DANGER_CRITICAL:
            w_survival, w_target = 1.0, 0.0
        elif danger > config.DANGER_MODERATE:
            w_survival, w_target = 0.85, 0.15
        else:
            w_survival, w_target = 0.3, 0.7

        force = w_survival * survival + w_target * target_force
        norm = np.linalg.norm(force)
        if norm < 1e-6:
            return 0.0, 0.0
        force /= norm
        return float(force[0]), float(force[1])
