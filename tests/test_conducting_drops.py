from math import pi

import pytest

from conductor_sim import combine_isolated_drops
from conductor_sim.model import EPSILON_0, FixedNetCharge, Grounded, SolidSphere


def drop(r, q):
    return SolidSphere(r, FixedNetCharge(q))


@pytest.mark.parametrize("q", [2e-12, -2e-12])
def test_workbook_page14_numerical_1(q):
    # Teacher SI inputs; the workbook gives only 27, r and common V1.
    r = 0.002
    run = combine_isolated_drops((drop(r, q),) * 27)
    v1 = q / (4 * pi * EPSILON_0 * r)
    for initial in run.initial:
        assert initial.potential_at((0, 0, 0)) == pytest.approx(v1, rel=1e-14)
    final = run.final
    assert final.radial_shells[0].radius_m == pytest.approx(3 * r)
    assert final.conductor_results[0].net_charge_c == pytest.approx(27 * q, abs=0)
    assert final.potential_at((0, 0, 0)) / v1 == pytest.approx(9)


@pytest.mark.parametrize("last_q", [-1e-9, 3e-9])
def test_unequal_signed_drops_compose_in_stages(last_q):
    drops = (drop(.01, 2e-9), drop(.02, -1e-9), drop(.03, last_q))
    epsilon = 2 * EPSILON_0
    def combine(group):
        return combine_isolated_drops(group, permittivity_f_per_m=epsilon).final
    direct = combine(drops)
    intermediate = combine_isolated_drops(drops[:2], permittivity_f_per_m=epsilon)
    staged = combine((intermediate, drops[2]))
    radius = (.01**3 + .02**3 + .03**3) ** (1 / 3)
    charge = 0.0 if last_q < 0 else 4e-9
    for state in (direct, staged):
        assert state.radial_shells[0].radius_m == pytest.approx(radius)
        assert state.conductor_results[0].net_charge_c == pytest.approx(charge, abs=0)
        assert state.potential_at((0, 0, 0)) == pytest.approx(charge / (4*pi*epsilon*radius), rel=2e-14, abs=0)
        assert state.electric_field_at((2*radius, 0, 0))[0] == pytest.approx(
            charge / (4*pi*epsilon*(2*radius)**2), rel=2e-14, abs=0)


def test_requires_multiple_isolated_drops():
    with pytest.raises(ValueError, match="at least two"):
        combine_isolated_drops((drop(.01, 0),))
    with pytest.raises(ValueError, match="fixed net charge"):
        combine_isolated_drops((drop(.01, 0), SolidSphere(.01, Grounded())))


def test_staged_charge_cancellation_retains_one_electron_magnitude():
    elementary_charge = 1.602176634e-19
    drops = tuple(drop(.25, q) for q in (.01, elementary_charge, -.01))
    pair = combine_isolated_drops(drops[:2])
    staged = combine_isolated_drops((pair, drops[2]))
    direct = combine_isolated_drops(drops)
    assert staged.initial[0] == pair.final
    assert staged.final == direct.final
    assert staged.final.conductor_results[0].net_charge_c == elementary_charge


@pytest.mark.parametrize("r,q,epsilon", [
    (1e4, 4e-320, EPSILON_0), (1e4, 1e-320, EPSILON_0), (1.0, 1.0, 1e308),
])
def test_numerically_unqualified_voltage_refused(r, q, epsilon):
    with pytest.raises(ValueError, match="numerical range"):
        combine_isolated_drops((drop(r, q),) * 2, permittivity_f_per_m=epsilon)


@pytest.mark.parametrize("r,q,epsilon", [(True, 1, EPSILON_0), (1, True, EPSILON_0), (1, 1, True)])
def test_boolean_si_values_refused(r, q, epsilon):
    with pytest.raises(ValueError, match="numerical"):
        combine_isolated_drops((drop(r, q),) * 2, permittivity_f_per_m=epsilon)


def test_inconsistent_stage_inventory_refused():
    from dataclasses import replace

    stage = combine_isolated_drops((drop(1, 1), drop(1, 2)))
    edited = replace(stage, constituents=(drop(1, 99), drop(1, 101)))
    with pytest.raises(ValueError, match="disagrees"):
        combine_isolated_drops((edited, drop(1, 1)))


def test_representable_field_at_large_probe_radius():
    state = combine_isolated_drops((drop(1e105, 1),) * 2).final
    distance = 2 * state.radial_shells[0].radius_m
    expected = 2 / (4*pi*EPSILON_0) / distance / distance
    assert state.electric_field_at((distance, 0, 0)) == pytest.approx(
        (expected, 0, 0), rel=1e-14, abs=0)
