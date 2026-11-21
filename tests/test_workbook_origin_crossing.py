from math import dist, hypot, isclose, pi, sqrt

from conductor_sim import EPSILON_0
from examples.workbook_origin_crossing import (
    first_downward_crossing, teacher_revision, workbook_scenario,
)


def test_printed_speed_at_origin():
    sample = first_downward_crossing(workbook_scenario())
    # Both source distances change from 5 m to 3 m; initial kinetic energy is zero.
    k = 1 / (4*pi*EPSILON_0)
    expected = sqrt(2 * 4e-6 * 4e-6 * k * (1/3 - 1/5) / .001)
    assert hypot(*sample.position_m) < 1e-8
    assert sample.velocity_m_s[1] < 0
    assert isclose(hypot(*sample.velocity_m_s), expected, rel_tol=1e-8)
    assert round(expected) == 6


def test_unequal_sources_shift_the_crossing():
    original = workbook_scenario()
    revision = teacher_revision(original)
    assert revision.parent_revision_id == original.revision_id
    assert len(revision.changes) == 1 and revision.changes[0].component_id == "q1"
    sample = first_downward_crossing(revision)
    x, y, z = sample.position_m
    assert x > .1 and abs(y) < 1e-8 and abs(z) < 1e-8
    # Independent work-energy check to the observed off-axis position.
    delta_v = sum(q * (1/dist(sample.position_m, center) - 1/5)
                  for q, center in ((2.1e-6, (3., 0., 0.)), (2e-6, (-3., 0., 0.))))
    expected = sqrt(2 * 4e-6 * delta_v / (4*pi*EPSILON_0*.001))
    assert isclose(hypot(*sample.velocity_m_s), expected, rel_tol=1e-8)
