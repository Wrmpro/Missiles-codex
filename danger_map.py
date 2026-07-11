"""Artificial potential field danger map."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

import config


class DangerMap:
    """Compute and query a grid of missile and boundary danger."""

    def __init__(self, width: int, height: int, resolution: int = config.GRID_RESOLUTION) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_w = max(1, int(np.ceil(width / resolution)))
        self.grid_h = max(1, int(np.ceil(height / resolution)))
        xs = np.arange(self.grid_w, dtype=np.float32) * resolution + resolution / 2.0
        ys = np.arange(self.grid_h, dtype=np.float32) * resolution + resolution / 2.0
        self.xx, self.yy = np.meshgrid(xs, ys)
        self._previous_grid: np.ndarray | None = None
        self._temporal_weights = self._build_temporal_weights()

    def _build_temporal_weights(self) -> np.ndarray:
        """Build deterministic temporal weights with stronger near-future danger."""

        weights = np.exp(-0.08 * np.arange(config.PREDICTION_STEPS, dtype=np.float32))
        horizons = (0.25, 0.5, 1.0, 2.0, 3.0)
        emphasis = (1.45, 1.30, 1.15, 1.0, 0.85)
        for horizon, factor in zip(horizons, emphasis):
            index = min(config.PREDICTION_STEPS - 1, int(horizon / max(config.PREDICTION_DT, 1e-6)))
            weights[index:] *= factor
        max_weight = float(weights.max())
        if max_weight > 0.0:
            weights /= max_weight
        return weights

    def compute(
        self,
        missile_trajectories: Dict[int, List[Tuple[float, float]]],
        player_pos: Tuple[float, float],
    ) -> np.ndarray:
        """Return a normalized danger grid in [0, 1]."""

        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for trajectory in missile_trajectories.values():
            for step, (mx, my) in enumerate(trajectory):
                dist = np.hypot(self.xx - mx, self.yy - my)
                safe_dist = np.maximum(dist, 1.0)
                influence = safe_dist < config.APF_D0
                discount = self._temporal_weights[min(step, len(self._temporal_weights) - 1)]
                potential = 0.5 * config.APF_ETA * ((1.0 / safe_dist) - (1.0 / config.APF_D0)) ** 2
                # Increase pressure in the immediate collision cone to improve short-horizon survival.
                critical_zone = np.where(safe_dist < 140.0, (140.0 - safe_dist) / 140.0, 0.0)
                grid += np.where(influence, potential * discount, 0.0) + 1.8 * critical_zone * discount

        margin = np.minimum.reduce(
            [self.xx, self.yy, self.width - self.xx, self.height - self.yy]
        )
        grid += np.exp(-np.maximum(margin, 0.0) / 40.0) * 0.35
        if self._previous_grid is not None:
            # Persist a fraction of prior danger to avoid fast oscillation between local minima.
            grid = 0.82 * grid + 0.18 * self._previous_grid
        grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=2.0, sigmaY=2.0)
        max_value = float(grid.max())
        if max_value > 0.0:
            grid /= max_value
        self._previous_grid = grid
        return grid

    def potential_at(self, grid: np.ndarray, x: float, y: float) -> float:
        """Return danger at a pixel coordinate."""

        gx = int(np.clip(x / self.resolution, 0, grid.shape[1] - 1))
        gy = int(np.clip(y / self.resolution, 0, grid.shape[0] - 1))
        return float(grid[gy, gx])

    def gradient_at(self, grid: np.ndarray, x: float, y: float, eps: float = 5.0) -> Tuple[float, float]:
        """Return numerical gradient of the danger field at a pixel coordinate."""

        dx = self.potential_at(grid, x + eps, y) - self.potential_at(grid, x - eps, y)
        dy = self.potential_at(grid, x, y + eps) - self.potential_at(grid, x, y - eps)
        return dx / (2.0 * eps), dy / (2.0 * eps)
