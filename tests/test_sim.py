import numpy as np
import pytest

from evolve import HAND, HI, LO, Geometry, simulate
from evolve.gait import convex_ccw, point_polygon_margin
from evolve.robot import leg_ik


def test_polygon_margin():
    sq = convex_ccw(np.array([[0, 0], [10, 0], [10, 10], [0, 10.0]]))
    assert point_polygon_margin(np.array([5.0, 5.0]), sq) == pytest.approx(5.0)
    assert point_polygon_margin(np.array([12.0, 5.0]), sq) < 0
    tri = convex_ccw(np.array([[0, 0], [10, 0], [0, 10.0]]))
    assert point_polygon_margin(np.array([1.0, 1.0]), tri) == pytest.approx(1.0)
    assert point_polygon_margin(np.array([1.0, 1.0]), np.array([[0, 0], [10, 0.0]])) < 0  # 2 feet: a line


def test_ik_neutral_stance_is_reachable_and_symmetric():
    g = Geometry()
    t1, t2, t3, ok = leg_ik(g, np.zeros(4), np.full(4, 55.0), np.full(4, -160.0))
    assert ok.all() and np.allclose(t1, 0) and np.allclose(t2, t2[0])
    _, _, _, ok = leg_ik(g, np.array([0.0]), np.array([55.0]), np.array([-400.0]))
    assert not ok[0]


def test_trot_cannot_be_statically_stable():
    e = simulate(HAND["05 trot"])
    assert e.margin <= 0.5  # two feet on the ground: the COM is at best on the diagonal line


def test_hand_gaits_are_reachable_but_tip_without_sway():
    for name in ("05 walk (crawl)", "05 trot"):
        e = simulate(HAND[name])
        assert e.reach_ok == 1.0 and e.margin < 0


def test_bounds_and_determinism():
    assert (LO < HI).all() and len(LO) == 10
    g = HAND["05 walk (crawl)"]
    assert simulate(g).fitness == simulate(g).fitness
