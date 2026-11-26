"""v0.3 studies: best gait per speed (where does static walking run out?) and the stability/effort
Pareto front. Also exports a gait for 05-quadruped."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cmaes import cmaes
from .gait import NAMES, simulate

LEG = ("FL", "FR", "RL", "RR")


def footfall(genome: np.ndarray) -> str:
    """Order in which feet touch down. Leg i touches down when (t/T + offset_i) wraps to 0, i.e. at
    t/T = (-offset_i) mod 1, so a LARGER offset means an EARLIER touchdown."""
    touchdown = (-np.array([0.0, *genome[:3]])) % 1.0
    return " → ".join(LEG[i] for i in np.argsort(touchdown))


def gait_name(genome: np.ndarray) -> str:
    off = np.array([0.0, *genome[:3]]) % 1.0
    d = lambda a, b: min(abs(a - b), 1 - abs(a - b))  # noqa: E731
    # check paired gaits first: tied offsets make the footfall order ambiguous
    if d(off[0], off[3]) < 0.1 and d(off[1], off[2]) < 0.1 and d(off[0], off[1]) > 0.35:
        return "trot"
    if d(off[0], off[2]) < 0.1 and d(off[1], off[3]) < 0.1 and d(off[0], off[1]) > 0.35:
        return "pace"
    order = footfall(genome)
    # Hildebrand: lateral sequence = each hind foot is followed by the fore foot on the SAME side
    # (LH, LF, RH, RF -> from LF: LF, RH, RF, LH); diagonal sequence = hind followed by the OPPOSITE fore foot
    if order == "FL → RR → FR → RL":
        return "lateral-sequence walk"
    if order == "FL → RL → FR → RR":
        return "diagonal-sequence walk"
    return "other (" + order + ")"


def _unit(genome: np.ndarray) -> np.ndarray:
    from .gait import HI, LO

    return (genome - LO) / (HI - LO)


def best_at_speed(speed: float, seeds=(0, 1, 2), budget: int = 960, w_effort: float = 2.0, warm: np.ndarray | None = None):
    """Several CMA-ES runs (random starts + an optional warm start from a neighbouring solution); keep the
    best. The servo-speed constraint is a cliff in the fitness landscape: random starts often land in the
    infeasible region, and continuation from a known-feasible gait gets out of it."""
    f = lambda g: simulate(g, speed=speed, w_effort=w_effort, samples=48).fitness  # noqa: E731
    runs = []
    for s in seeds:
        h = cmaes(f, budget=budget, seed=s)
        runs.append((h.best[-1], h.best_genome, h))
    if warm is not None:
        h = cmaes(f, budget=budget, seed=99, x0=_unit(warm), sigma0=0.1)
        runs.append((h.best[-1], h.best_genome, h))
    scored = [(simulate(g, speed=speed, w_effort=w_effort).fitness, g, h) for _, g, h in runs]
    fbest, g, h = max(scored, key=lambda r: r[0])
    return g, simulate(g, speed=speed), h


def speed_sweep(speeds=(30, 45, 60, 80, 100, 130, 160, 200), **kw) -> list[dict]:
    found, warm = {}, None
    for v in speeds:
        g, _, _ = best_at_speed(v, warm=warm, **kw)
        found[v], warm = g, g
    # solution sharing: a gait found at one speed may beat the optimizer's pick at another
    pool = list(found.values())
    rows = []
    for v in speeds:
        g = max(pool, key=lambda p: simulate(p, speed=v).fitness)
        e = simulate(g, speed=v)
        rows.append({"speed": v, "margin": e.margin, "effort": e.effort, "reach": e.reach_ok, "speed_ok": e.speed_ok,
                     "peak_joint_speed": e.peak_joint_speed, "gait": gait_name(g), "from_pool": bool(g is not found[v]),
                     "footfall": footfall(g), **{n: float(x) for n, x in zip(NAMES, g)}})
    return rows


def pareto(speed: float = 45.0, weights=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)) -> list[dict]:
    found, warm = {}, None
    for w in weights:
        g, _, _ = best_at_speed(speed, seeds=(0, 1), w_effort=w, warm=warm)
        found[w], warm = g, g
    pool = list(found.values())
    pts = []
    for w in weights:
        g = max(pool, key=lambda p: simulate(p, speed=speed, w_effort=w).fitness)
        e = simulate(g, speed=speed)
        pts.append({"w_effort": w, "margin": e.margin, "effort": e.effort, "speed_ok": e.speed_ok, "genome": g.tolist(),
                    "gait": gait_name(g)})
    return pts


def export_05(genome: np.ndarray, speed: float) -> str:
    """A snippet for 05-quadruped: add the gait to GAITS in quadruped/gait.py and set the stance/gait config."""
    g = genome
    offsets = {"FL": 0.0, "FR": round(float(g[0]), 3), "RL": round(float(g[1]), 3), "RR": round(float(g[2]), 3)}
    return (f'# evolved by 13-gait-evolution for {speed:.0f} mm/s ({gait_name(g)})\n'
            f'# quadruped/gait.py -> GAITS["evolved"] = ({json.dumps(offsets)}, {g[3]:.3f})\n'
            f'# config/robot.yaml:\n'
            f'stance:\n  height: {g[7]:.0f}\n  foot_y: {55 + (g[6] - 55):.0f}\n'
            f'gait:\n  period: {g[4]:.2f}\n  step_height: {g[5]:.0f}\n' +
            (f'# body sway: {g[8]:.0f} mm at phase {g[9]:.2f} (needs Pose shifting in Robot.tick)\n' if g[8] >= 1 else '# body sway: none needed\n'))


def main() -> int:
    rows = speed_sweep()
    for r in rows:
        print(f"{r['speed']:4.0f} mm/s  margin {r['margin']:6.1f}  effort {r['effort']:5.2f}  duty {r['duty']:.2f}  T {r['period']:.2f}  "
              f"step {r['step_h']:.0f}  sway {r['sway']:4.1f}  peak {r['peak_joint_speed']:.1f} rad/s ok {r['speed_ok']:.2f}  {r['gait']}")
    pts = pareto(speed=60.0)
    for p in pts:
        print(f"w={p['w_effort']:5.1f}  margin {p['margin']:6.1f}  effort {p['effort']:5.2f}  ok {p['speed_ok']:.2f}  {p['gait']}")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/study.json").write_text(json.dumps({"speed_sweep": rows, "pareto": pts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
