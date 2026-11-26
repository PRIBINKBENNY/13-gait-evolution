import numpy as np
import pytest
import yaml

from evolve import HAND, simulate
from evolve.cmaes import cmaes
from evolve.gait import PENALTY
from evolve.robot import Geometry
from evolve.study import export_05, gait_name


def test_servo_speed_limit_flags_05s_shipped_walk():
    e = simulate(HAND["05 walk (crawl)"])
    assert e.peak_joint_speed > Geometry().max_joint_speed * 1.8   # ~11 rad/s vs MG996R ~5.2
    assert e.speed_ok < 0.5 and e.fitness < -PENALTY * 0.4


def test_feasibility_first():
    slow = HAND["05 walk (crawl)"].copy()
    slow[4] = 1.6                          # same walk, 3x slower cycle
    assert simulate(slow).speed_ok > simulate(HAND["05 walk (crawl)"]).speed_ok
    feasible_bad_margin = -60.0 - 2 * 30.0                 # worst feasible score: margin clamps at -60
    infeasible_great_margin = 60.0 - 2 * 0.0 - PENALTY / 63  # best imaginable score with ONE violating sample
    assert feasible_bad_margin > infeasible_great_margin


@pytest.fixture(scope="module")
def evolved_45():
    from evolve.gait import NAMES
    from evolve.study import speed_sweep

    rows = speed_sweep(speeds=(30, 45))  # the study's pipeline: restarts + warm start from 30 mm/s + sharing
    r = rows[1]
    g = np.array([r[n] for n in NAMES])
    return g, simulate(g, speed=45)


def test_evolves_a_feasible_stable_gait_at_walking_pace(evolved_45):
    g, e = evolved_45
    assert e.speed_ok == 1.0 and e.reach_ok == 1.0 and e.margin > 0


def test_rediscovers_the_lateral_sequence_walk(evolved_45):
    assert gait_name(evolved_45[0]) == "lateral-sequence walk"


def test_gait_names():
    assert gait_name(HAND["05 trot"]) == "trot"
    assert gait_name(HAND["05 walk (crawl)"]) == "diagonal-sequence walk"
    assert gait_name(np.array([0.5, 0.25, 0.75, 0.8, 1, 30, 55, 160, 0, 0])) == "lateral-sequence walk"


def test_export_is_valid_yaml_for_05():
    txt = export_05(np.array([0.55, 0.2, 0.75, 0.85, 1.6, 25, 55, 160, 0, 0]), 80)
    cfg = yaml.safe_load(txt)
    assert cfg["gait"]["period"] == 1.6 and cfg["stance"]["height"] == 160
