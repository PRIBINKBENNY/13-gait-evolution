import numpy as np

from evolve import simulate
from evolve.cmaes import cmaes, random_search, to_genome
from evolve.gait import HI, LO


def test_cmaes_solves_a_shifted_sphere():
    target = np.full(10, 0.3)
    f = lambda g: -float(np.sum(((g - LO) / (HI - LO) - target) ** 2))  # noqa: E731
    h = cmaes(f, budget=1600, seed=1)
    assert h.best[-1] > -1e-3
    assert all(b >= a for a, b in zip(h.best, h.best[1:]))  # best-so-far never gets worse


def test_phase_genes_wrap_instead_of_clipping():
    g = to_genome(np.array([1.25, -0.25, 0.5, 2.0, -1.0, 0.5, 0.5, 0.5, 0.5, 1.75]))
    assert np.isclose(g[0], 0.25) and np.isclose(g[1], 0.75) and np.isclose(g[9], 0.75)
    assert g[3] == HI[3] and g[4] == LO[4]  # non-phase genes clip to bounds


def test_cmaes_beats_random_search_at_equal_budget():
    f = lambda g: simulate(g, samples=32).fitness  # noqa: E731
    c = [cmaes(f, budget=320, seed=s).best[-1] for s in range(2)]
    r = [random_search(f, budget=320, seed=s).best[-1] for s in range(2)]
    assert np.mean(c) > np.mean(r) + 5


def test_evolved_gait_is_stable_and_rediscovers_lateral_sequence():
    h = cmaes(lambda g: simulate(g).fitness, budget=640, seed=4)
    g = h.best_genome
    e = simulate(g)
    assert e.margin > 0 and e.reach_ok == 1.0
    order = np.argsort([0.0, g[0], g[1], g[2]])  # legs FL, FR, RL, RR by phase
    assert [("FL", "FR", "RL", "RR")[i] for i in order] == ["FL", "RL", "FR", "RR"]
