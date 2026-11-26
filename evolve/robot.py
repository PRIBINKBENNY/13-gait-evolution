"""Leg geometry and vectorised inverse kinematics for the 12-servo quadruped (same model as 05-quadruped)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
SIDE = np.array([1, -1, 1, -1])        # +1 left, -1 right
FRONT = np.array([1, 1, -1, -1])       # +1 front, -1 rear


@dataclass(frozen=True)
class Geometry:
    body_length: float = 186.0
    body_width: float = 78.0
    coxa: float = 55.0
    femur: float = 107.0
    tibia: float = 130.0
    knee_direction: int = -1
    # joint limits (deg), as in 05's config
    hip_lim: tuple = (-35.0, 35.0)
    femur_lim: tuple = (-70.0, 80.0)
    knee_lim: tuple = (0.0, 140.0)
    # MG996R: ~0.17 s/60 deg unloaded at 6 V (6.2 rad/s); ~0.2 s/60 deg under load -> 5.2 rad/s
    max_joint_speed: float = 5.2

    def hips(self) -> np.ndarray:
        """[4, 2] hip positions in the body frame (x forward, y left), mm."""
        return np.column_stack([FRONT * self.body_length / 2, SIDE * self.body_width / 2])


def leg_ik(g: Geometry, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Vectorised IK in each leg's frame (y outward). Returns (hip, femur, knee) in rad and a validity mask."""
    d2 = y * y + z * z
    ok = d2 >= g.coxa ** 2
    h = np.sqrt(np.maximum(d2 - g.coxa ** 2, 1e-9))
    t1 = np.arctan2(y, -z) - np.arctan2(g.coxa, h)
    r2 = x * x + h * h
    c3 = (r2 - g.femur ** 2 - g.tibia ** 2) / (2 * g.femur * g.tibia)
    ok &= (c3 <= 1) & (c3 >= -1)
    c3 = np.clip(c3, -1, 1)
    t3 = np.arccos(c3)
    s = g.knee_direction
    alpha = np.arctan2(g.tibia * np.sin(t3), g.femur + g.tibia * np.cos(t3))
    t2 = np.arctan2(x, h) - s * alpha
    lim = lambda a, l: (np.degrees(a) >= l[0]) & (np.degrees(a) <= l[1])  # noqa: E731
    ok &= lim(t1, g.hip_lim) & lim(t2, g.femur_lim) & lim(t3, g.knee_lim)
    return t1, t2, t3, ok
