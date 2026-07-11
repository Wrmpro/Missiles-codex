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
                discount = np.exp(-step * 0.08)
                potential = 0.5 * config.APF_ETA * ((1.0 / safe_dist) - (1.0 / config.APF_D0)) ** 2
                grid += np.where(influence, potential * discount, 0.0)

        margin = np.minimum.reduce(
            [self.xx, self.yy, self.width - self.xx, self.height - self.yy]
        )
        grid += np.exp(-np.maximum(margin, 0.0) / 40.0) * 0.35
        grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=2.0, sigmaY=2.0)
        max_value = float(grid.max())
        if max_value > 0.0:
            grid /= max_value
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
