"""Figures + GIF for the README:  python -m evolve.viz  ->  docs/"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .cmaes import cmaes, random_search
from .gait import HAND, NAMES, convex_ccw, simulate
from .robot import Geometry

LEGS = ("FL", "FR", "RL", "RR")
GOOD, BAD, INK, MUTED = "#2a9d8f", "#e63946", "#1d3557", "#adb5bd"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _study():
    d = json.loads(Path("docs/study.json").read_text())
    return d, {r["speed"]: np.array([r[n] for n in NAMES]) for r in d["speed_sweep"]}


def _draw_top(ax, frame, title):
    g = Geometry()
    hips = g.hips()
    bx, by = frame["com"]
    body = np.array([[g.body_length / 2, g.body_width / 2], [g.body_length / 2, -g.body_width / 2],
                     [-g.body_length / 2, -g.body_width / 2], [-g.body_length / 2, g.body_width / 2]]) + [bx, by]
    ax.fill(body[:, 0], body[:, 1], color="#f2a541", alpha=0.35, lw=0)
    feet, st = frame["feet"], frame["stance"]
    sup = feet[st, :2]
    ok = frame["margin"] > 0
    if len(sup) >= 3:
        poly = convex_ccw(sup)
        ax.fill(poly[:, 0], poly[:, 1], color=GOOD if ok else BAD, alpha=0.18, lw=0)
        ax.plot(*np.vstack([poly, poly[:1]]).T, color=GOOD if ok else BAD, lw=1.5)
    elif len(sup) == 2:
        ax.plot(sup[:, 0], sup[:, 1], color=BAD, lw=2.5)
    for i in range(4):
        hx, hy = hips[i] + [bx, by]
        ax.plot([hx, feet[i, 0]], [hy, feet[i, 1]], color=INK, lw=1.2, alpha=0.6)
        ax.plot(feet[i, 0], feet[i, 1], "o", ms=9 if st[i] else 6, color=INK if st[i] else "white", mec=INK, mew=1.5)
        ax.text(feet[i, 0], feet[i, 1] + 14, LEGS[i], ha="center", fontsize=7, color=MUTED)
    ax.plot(bx, by, "X", ms=12, color=GOOD if ok else BAD, mec="black", mew=0.8, zorder=5)
    ax.set_xlim(-230, 230)
    ax.set_ylim(-190, 190)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10)
    ax.text(0.02, 0.03, f"margin {frame['margin']:+.0f} mm" if frame["margin"] > -500 else "margin: 2 feet (line)",
            transform=ax.transAxes, color=GOOD if ok else BAD, weight="bold")


def gif(out: Path):
    plt = _plt()
    from PIL import Image

    _, best = _study()
    hand = simulate(HAND["05 walk (crawl)"], speed=45, record=True, samples=48).details["frames"]
    evo_g = best[45]
    evo = simulate(evo_g, speed=45, record=True, samples=48).details["frames"]
    # resample both to the same number of frames over one cycle each
    def cyc(frames):
        n = len(frames) // 2
        return [frames[int(i * n / 48)] for i in range(48)]

    imgs = []
    for fh, fe in zip(cyc(hand), cyc(evo)):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
        _draw_top(axes[0], fh, "05's hand-written walk (diagonal sequence)")
        _draw_top(axes[1], fe, "evolved: lateral-sequence walk")
        fig.text(0.5, 0.02, "● foot on ground  ○ swinging  ✕ centre of mass  · shaded = support polygon (red = would tip)",
                 ha="center", fontsize=8, color="#555")
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        buf = io.BytesIO()
        fig.savefig(buf, dpi=66, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        imgs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=48))
    imgs[0].save(out / "stability.gif", save_all=True, append_images=imgs[1:], duration=110, loop=0, optimize=True)


def footfalls(out: Path):
    plt = _plt()
    _, best = _study()
    gaits = [("05 hand walk", HAND["05 walk (crawl)"]), ("05 hand trot", HAND["05 trot"]),
             ("evolved @ 30 mm/s", best[30]), ("evolved @ 60 mm/s", best[60]), ("evolved @ 100 mm/s", best[100]),
             ("evolved @ 160 mm/s", best[160])]
    fig, axes = plt.subplots(len(gaits), 1, figsize=(9, 1.05 * len(gaits) + 0.6), sharex=True)
    for ax, (name, g) in zip(axes, gaits):
        offs = [0.0, g[0], g[1], g[2]]
        duty = g[3]
        for i, o in enumerate(offs):
            for c in range(2):
                start = (c - o) % 2 if False else c + ((-o) % 1.0)
                for s0 in (start - 1, start):
                    ax.barh(i, duty, left=s0, height=0.6, color=INK, alpha=0.85)
        ax.set_yticks(range(4), LEGS, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 2)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        for s in ("left",):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("gait cycles (bars = foot on the ground)")
    fig.suptitle("Footfall diagrams: hand-written vs evolved", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(out / "footfalls.png", dpi=100, facecolor="white")
    plt.close(fig)


def speed_chart(out: Path):
    plt = _plt()
    d, _ = _study()
    rows = [r for r in d["speed_sweep"] if r["speed"] <= 160]
    v = [r["speed"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    m = [max(r["margin"], -40) for r in rows]
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].plot(v, m, "o-", color=GOOD, lw=2)
    for r, y in zip(rows, m):
        if r["gait"] == "trot":
            axes[0].annotate("trot", (r["speed"], y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=BAD)
    axes[0].axvspan(74, 165, color=BAD, alpha=0.06)
    axes[0].text(78, max(m) * 0.8, "beyond static stability\n(needs dynamic balance)", fontsize=8, color=BAD)
    axes[0].set(xlabel="forward speed (mm/s)", ylabel="worst-case margin (mm)", title="Best stability margin found at each speed")
    axes[1].plot(v, [r["duty"] for r in rows], "o-", color=INK, lw=2, label="duty factor")
    axes[1].set(xlabel="forward speed (mm/s)", ylabel="duty factor", title="Duty factor falls with speed (as in animals)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "speed_study.png", dpi=105, facecolor="white")
    plt.close(fig)


def convergence(out: Path, seeds=(0, 1, 2)):
    plt = _plt()
    f = lambda g: simulate(g, speed=45, samples=48).fitness  # noqa: E731
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for name, fn, col in (("random search", lambda s: random_search(f, 960, s), BAD), ("CMA-ES", lambda s: cmaes(f, budget=960, seed=s), GOOD)):
        curves = [fn(s) for s in seeds]
        for h in curves:
            ax.plot(h.evals, np.clip(h.best, -80, None), color=col, alpha=0.35, lw=1)
        ax.plot(curves[0].evals, np.median([np.clip(h.best, -80, None) for h in curves], 0), color=col, lw=2.5, label=name)
    ax.set(xlabel="evaluations", ylabel="best fitness so far", title="Search at 45 mm/s: CMA-ES vs random (3 seeds)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=105, facecolor="white")
    plt.close(fig)


def main() -> int:
    out = Path("docs")
    gif(out)
    footfalls(out)
    speed_chart(out)
    convergence(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
