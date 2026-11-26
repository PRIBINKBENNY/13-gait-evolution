"""A parameterised gait (the genome) and its quasi-static evaluation.

Genome (10 numbers, each with bounds):
  off_FR, off_RL, off_RR   phase offsets of the other legs relative to FL (cycles)
  duty                     fraction of the cycle each foot is on the ground
  period                   seconds per cycle
  step_h                   swing foot lift (mm)
  width                    sideways foot placement from the hip (mm)
  height                   body height (mm)
  sway                     amplitude of body shift towards the supporting legs (mm)
  sway_phase               timing of that shift (cycles)

Quasi-static means: at every instant the centre of mass must sit inside the polygon of feet on the
ground (the robot would not tip over if it froze). That is the right model for a slow, servo-driven
walker; it is NOT a model of dynamic trotting, which balances by momentum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot import FRONT, SIDE, Geometry, leg_ik

NAMES = ["off_FR", "off_RL", "off_RR", "duty", "period", "step_h", "width", "height", "sway", "sway_phase"]
LO = np.array([0.0, 0.0, 0.0, 0.50, 0.40, 10.0, 40.0, 120.0, 0.0, 0.0])
HI = np.array([1.0, 1.0, 1.0, 0.95, 1.60, 60.0, 90.0, 200.0, 45.0, 1.0])

# hand-designed references (what you'd write by hand, and what 05-quadruped ships with)
HAND = {
    "05 walk (crawl)": np.array([0.50, 0.75, 0.25, 0.78, 0.50, 35.0, 55.0, 160.0, 0.0, 0.0]),
    "05 trot": np.array([0.50, 0.50, 0.00, 0.55, 0.50, 35.0, 55.0, 160.0, 0.0, 0.0]),
    "textbook crawl + sway": np.array([0.50, 0.75, 0.25, 0.80, 1.00, 30.0, 60.0, 160.0, 20.0, 0.375]),
}


@dataclass
class Eval:
    fitness: float
    margin: float          # worst-case stability margin over the cycle (mm); < 0 = would tip
    effort: float          # mean summed joint speed (rad/s): servo work and wear
    reach_ok: float        # fraction of samples where every leg pose is reachable within joint limits
    speed: float
    details: dict | None = None


def point_polygon_margin(p: np.ndarray, poly: np.ndarray) -> float:
    """Signed distance from p to the convex polygon boundary (+ inside). poly: [k,2] in CCW order."""
    k = len(poly)
    if k < 3:
        if k == 2:  # on a line: margin is minus the distance to that segment (never statically stable)
            a, b = poly
            t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
            return -float(np.linalg.norm(p - (a + t * (b - a))))
        return -1e3
    d = []
    inside = True
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        e = b - a
        n = np.array([-e[1], e[0]]) / max(np.linalg.norm(e), 1e-9)  # inward normal for CCW
        s = float(np.dot(p - a, n))
        d.append(s)
        inside &= s >= 0
    return min(d) if inside else min(d)


def convex_ccw(points: np.ndarray) -> np.ndarray:
    """Order 3-4 support points counter-clockwise around their centroid."""
    c = points.mean(0)
    ang = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
    return points[np.argsort(ang)]


def simulate(genome: np.ndarray, speed: float = 80.0, g: Geometry = Geometry(), samples: int = 64,
             cycles: int = 2, record: bool = False) -> Eval:
    off_fr, off_rl, off_rr, duty, period, step_h, width, height, sway, sway_ph = genome
    offsets = np.array([0.0, off_fr, off_rl, off_rr])
    hips = g.hips()
    neutral = hips + np.column_stack([np.zeros(4), SIDE * (g.coxa + (width - 55.0))])
    stride = speed * period * duty                       # foot travel while on the ground
    n = samples * cycles
    ts = np.arange(n) / samples * period
    margins, speeds, ok_all = [], [], []
    prev = None
    frames = [] if record else None
    for t in ts:
        ph = (t / period + offsets) % 1.0
        stance = ph < duty
        u = np.where(stance, ph / duty, (ph - duty) / (1 - duty))
        k = np.where(stance, 0.5 - u, -0.5 + (1 - np.cos(np.pi * u)) / 2)
        fx = neutral[:, 0] + stride * k
        fy = neutral[:, 1]
        fz = np.where(stance, -height, -height + step_h * np.sin(np.pi * u))
        # body sway: shift the body (and so the COM) in a loop, phase-locked to the gait
        a = 2 * np.pi * (t / period + sway_ph)
        bx, by = sway * 0.5 * np.cos(2 * a), sway * np.sin(a)
        # foot positions relative to the shifted body -> each leg's frame
        lx = fx - bx - hips[:, 0]
        ly = (fy - by - hips[:, 1]) * SIDE
        lz = fz
        t1, t2, t3, ok = leg_ik(g, lx, ly, lz)
        ok_all.append(ok.all())
        q = np.concatenate([t1, t2, t3])
        if prev is not None:
            speeds.append(np.abs(q - prev).sum() / (period / samples))
        prev = q
        sup = np.column_stack([fx[stance], fy[stance]])
        com = np.array([bx, by])
        m = point_polygon_margin(com, convex_ccw(sup)) if stance.sum() >= 3 else (
            point_polygon_margin(com, sup) if stance.sum() == 2 else -1e3)
        margins.append(m)
        if record:
            frames.append({"t": t, "feet": np.column_stack([fx, fy, fz]), "stance": stance.copy(), "com": com,
                           "margin": m, "ok": bool(ok.all())})
    margins, ok_frac = np.array(margins), float(np.mean(ok_all))
    margin = float(margins.min())
    effort = float(np.mean(speeds))
    fit = fitness(margin, effort, ok_frac, stance_frac=duty)
    return Eval(fit, margin, effort, ok_frac, speed, {"frames": frames, "margins": margins} if record else None)


def fitness(margin: float, effort: float, ok_frac: float, stance_frac: float, w_effort: float = 2.0) -> float:
    """Maximise worst-case stability margin (mm), minus a servo-effort cost, with a heavy penalty for
    any unreachable pose (a gait the robot can't physically execute is worthless)."""
    return min(margin, 60.0) - w_effort * effort - 400.0 * (1 - ok_frac)


def clip(genome: np.ndarray) -> np.ndarray:
    return np.clip(genome, LO, HI)


def random_genome(rng: np.random.Generator) -> np.ndarray:
    return LO + rng.random(len(LO)) * (HI - LO)
