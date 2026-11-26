# 13 · Gait evolution for a 12-servo robot dog

Instead of hand-writing gaits, **evolve** them. Search over leg timing, stance and body sway for gaits that never tip over (quasi-statically) and work the servos as little as possible, on the exact leg geometry of [05-quadruped](https://github.com/PRIBINKBENNY/05-quadruped).

**Status: v0.2, CMA-ES.**

## v0.1: a stability simulator and baselines

A gait is a **10-number genome**: phase offsets of three legs relative to front-left, duty factor, period, step height, stance width, body height, and **body sway** (amplitude and timing of shifting the body over the supporting legs).

Each genome is run through the real leg IK with joint limits, and scored on:
- **worst-case stability margin**: how far the centre of mass stays inside the polygon of feet on the ground, at the worst moment of the cycle (negative = it would tip if it froze)
- **servo effort**: summed joint speeds, a proxy for energy and wear
- **reachability**: any pose outside joint limits is heavily penalized

At 80 mm/s:

| gait | worst-case margin | effort (rad/s) | fitness |
|---|---|---|---|
| 05 walk (crawl), hand-written | −6.5 mm | 12.1 | −30.6 |
| 05 trot, hand-written | −6.0 mm | 11.3 | −28.6 |
| "textbook" crawl + sway guess | −21.4 mm | 10.5 | −42.5 |
| random search, 640 evaluations (3 seeds) | −3.9 / +4.0 / −2.9 mm | 6.7–9.3 | −17.3 / −14.6 / −17.4 |

Two findings already: every hand-written gait **would tip** at some moment, and a guessed body sway made it **worse**, because the sway timing matters.

```bash
pip install -e ".[dev]" && pytest
```

## v0.2: CMA-ES rediscovers how animals walk

A from-scratch **CMA-ES** (covariance matrix adaptation evolution strategy, about 100 lines of NumPy). It adapts the shape of its search distribution, so it learns which genes move together. Phase genes wrap around (0 and 1 are the same moment) instead of being clipped.

Same budget, 640 evaluations, 5 seeds:

| | best fitness | worst-case margin | effort |
|---|---|---|---|
| random search | −14 to −20 | **tips on 4 of 5** (−3.9 to +4.0 mm) | 6.0–9.3 |
| **CMA-ES** | **−6.3 to +2.4** | **stable on 5 of 5** (+5.6 to +13 mm) | **5.1–6.3** |

**All five independent runs converged on the same footfall order: FL → RL → FR → RR**, the **lateral-sequence walk**. That's the pattern quadruped animals use for slow walking, and the one McGhee & Frank (1968) proved maximizes static stability. The hand-written gait in 05 uses the diagonal sequence (FL → RR → FR → RL), which is why it tips.

**What the optimizer exploited (not hidden):** it pushes duty to the maximum (0.95), the period to the slowest allowed (about 1.6 s), and **foot lift to 10 mm**, which would trip on a rug. Body sway goes unused because with 95% duty four feet are down most of the time. The model rewards exactly what it was told to, so v0.3 tightens the problem.
