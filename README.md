# 13 · Gait evolution for a 12-servo robot dog

Instead of hand-writing gaits, **evolve** them. Search over leg timing, stance and body sway for gaits that never tip over (quasi-statically) and work the servos as little as possible, on the exact leg geometry of [05-quadruped](https://github.com/PRIBINKBENNY/05-quadruped).

**Status: v0.1, simulator + baselines.**

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
