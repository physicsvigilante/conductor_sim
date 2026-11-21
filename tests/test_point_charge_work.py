from fractions import Fraction
from math import nextafter, pi, sqrt

import pytest

from conductor_sim import (EPSILON_0, ElectrostaticProblem, Grounded, PointCharge,
                           SolidSphere, rearrange_point_charges)


def test_workbook_p14_q5_and_teacher_change():
    # Three diagonals of equal length. All squared distances match exactly.
    vertices = ((1, 1, 0), (1, 0, 1), (0, 1, 1))
    twice = tuple(tuple(2*x for x in point) for point in vertices)
    def expansion(charges):
        model = ElectrostaticProblem(tuple(PointCharge(q, p) for q, p in zip(charges, vertices)))
        return rearrange_point_charges(model, twice)
    assert expansion((-2e-9, -2e-9, 1e-9)).external_work_j == 0.0
    changed = expansion((-2e-9, -2e-9, 2e-9))
    expected = 2e-18 / (4*pi*EPSILON_0*sqrt(2))
    assert changed.external_work_j == pytest.approx(expected, rel=1e-14, abs=0)
    assert changed.field_work_j == -changed.external_work_j


def test_composed_stages_reverse_and_medium():
    initial = ElectrostaticProblem((PointCharge(1e-9), PointCharge(3e-9, (1,0,0))),
                                   permittivity_f_per_m=3*EPSILON_0)
    intermediate = rearrange_point_charges(initial, ((0,0,0), (2,0,0)))
    final = ((0,0,0), (4,0,0))
    second = rearrange_point_charges(intermediate.final, final)
    direct = rearrange_point_charges(initial, final)
    expected = 3e-18*(1/4-1)/(4*pi*3*EPSILON_0)
    assert direct.external_work_j == pytest.approx(expected, rel=2e-15, abs=0)
    assert intermediate.external_work_j + second.external_work_j == pytest.approx(expected, rel=2e-15, abs=0)
    reverse = rearrange_point_charges(direct.final, tuple(p.position_m for p in initial.charges))
    assert reverse.external_work_j == -direct.external_work_j
    assert reverse.final == initial


def test_ulps_in_geometry_resolve_small_work():
    end = nextafter(3., 4.)
    case = ElectrostaticProblem((PointCharge(1.), PointCharge(-1., (3.,0.,0.))))
    run = rearrange_point_charges(case, ((0,0,0), (end,0,0)))
    expected = (Fraction(1,3) - 1/Fraction(end)) / (Fraction(4*pi)*Fraction(EPSILON_0))
    assert run.external_work_j == float(expected)


def test_reject_singular_endpoints_and_conductors():
    a = PointCharge(1)
    b = PointCharge(-1,(1,0,0))
    for initial, end in (((a,a),((0,0,0),(1,0,0))), ((a,b),((0,0,0),(0,0,0)))):
        with pytest.raises(ValueError, match="singular"):
            rearrange_point_charges(ElectrostaticProblem(initial), end)
    with pytest.raises(ValueError, match="responding conductors"):
        rearrange_point_charges(ElectrostaticProblem(conductors=(SolidSphere(1,Grounded()),)), ())


@pytest.mark.parametrize("direction", [(1,0,0), (1,1,0)])
def test_collinear_exact_cancellation_at_distinct_pair_distances(direction):
    # At separations d,d,2d the weights are -4,-4,+16.
    def points(scale):
        return tuple(tuple(scale*i*x for x in direction) for i in (0,1,2))
    initial = ElectrostaticProblem(tuple(PointCharge(q,p) for q,p in zip((4,-1,4),points(3))))
    assert rearrange_point_charges(initial, points(5)).external_work_j == 0


def test_rational_reciprocal_rounds_exact_halfway_to_even():
    initial = ElectrostaticProblem((PointCharge(4*pi), PointCharge(3., (3.,0.,0.))),
                                   permittivity_f_per_m=1.)
    result = rearrange_point_charges(initial, ((0,0,0), (3.*2**54,0,0)))
    # Exact work -1 + 2^-54 is halfway between -1 and its neighbour toward zero.
    assert result.external_work_j == -1.
