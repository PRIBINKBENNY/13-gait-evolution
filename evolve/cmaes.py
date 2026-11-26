"""CMA-ES (Hansen's (mu/mu_w, lambda) evolution strategy with covariance matrix adaptation) in NumPy.

Search happens in the unit cube; genes that are phases (leg offsets, sway timing) wrap around
instead of being clipped, since 1.0 and 0.0 are the same moment in the cycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .gait import HI, LO

PERIODIC = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1], bool)


def to_genome(z: np.ndarray) -> np.ndarray:
    u = np.where(PERIODIC, z % 1.0, np.clip(z, 0.0, 1.0))
    return LO + u * (HI - LO)


@dataclass
class History:
    evals: list[int] = field(default_factory=list)
    best: list[float] = field(default_factory=list)
    mean_fit: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    best_genome: np.ndarray | None = None


def cmaes(f: Callable[[np.ndarray], float], n: int = 10, budget: int = 640, popsize: int = 16, sigma0: float = 0.3,
          x0: np.ndarray | None = None, seed: int = 0) -> History:
    """Maximise f(genome)."""
    rng = np.random.default_rng(seed)
    lam = popsize
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    m = rng.random(n) if x0 is None else x0.copy()
    sigma = sigma0
    C = np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    B, D = np.eye(n), np.ones(n)
    hist = History()
    best_f, evals, gen = -np.inf, 0, 0
    while evals + lam <= budget:
        gen += 1
        Z = rng.standard_normal((lam, n))
        Y = Z @ (B * D).T
        X = m + sigma * Y
        F = np.array([f(to_genome(x)) for x in X])
        evals += lam
        order = np.argsort(-F)
        if F[order[0]] > best_f:
            best_f, hist.best_genome = float(F[order[0]]), to_genome(X[order[0]])
        Ysel = Y[order[:mu]]
        y_w = w @ Ysel
        m = m + sigma * y_w
        # step-size path and covariance paths
        invsqrtC = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y_w)
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (Ysel.T * w) @ Ysel)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = float(np.clip(sigma, 1e-4, 1.0))
        C = (C + C.T) / 2
        evals_d, Bv = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(evals_d, 1e-20))
        B = Bv
        hist.evals.append(evals)
        hist.best.append(best_f)
        hist.mean_fit.append(float(F.mean()))
        hist.sigma.append(sigma)
    return hist


def random_search(f: Callable[[np.ndarray], float], budget: int = 640, seed: int = 0, batch: int = 16) -> History:
    rng = np.random.default_rng(seed)
    hist, best_f = History(), -np.inf
    for e in range(0, budget, batch):
        for _ in range(batch):
            g = to_genome(rng.random(10))
            v = f(g)
            if v > best_f:
                best_f, hist.best_genome = v, g
        hist.evals.append(e + batch)
        hist.best.append(best_f)
    return hist
