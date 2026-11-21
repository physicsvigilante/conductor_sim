from math import nextafter, sqrt

import pytest

from conductor_sim import ConnectionPlan, ElectricalLink, ElectrostaticProblem, FixedNetCharge, SolidSphere
from conductor_sim.distant_spheres import ground_and_disconnect_distant_sphere
from conductor_sim.solver import UnsupportedProblem


def sphere(name, radius, q, center):
    return SolidSphere(radius, FixedNetCharge(q), center, name)


def switch(problem, name):
    plan = ConnectionPlan(problem, (ElectricalLink((name,), "grounded"),))
    return ground_and_disconnect_distant_sphere(plan, approximation="centre-monopole")


def test_workbook_q31_disconnect_b_before_grounding_c():
    a, d, q = 1., 100., 2e-8
    initial = ElectrostaticProblem(conductors=(sphere("A", a, q/2, (0, 0, 0)),
        sphere("B", a, 0, (d, 0, 0)), sphere("C", a, 0, (d/2, sqrt(3)*d/2, 0))))
    after_b = switch(initial, "B")
    after_c = switch(after_b, "C")
    assert after_c.conductors[1] is after_b.conductors[1]
    assert after_c.conductors[0] is initial.conductors[0]
    qb, qc = (c.constraint.charge_c for c in after_c.conductors[1:])
    assert qb == pytest.approx(-q*a/(2*d), rel=1e-14, abs=0)
    assert qc == pytest.approx(-q*a*(d-a)/(2*d*d), rel=1e-14, abs=0)
    assert switch(initial, "C").conductors[2].constraint.charge_c != qc
    assert initial.conductors[1].constraint.charge_c == 0


def test_unequal_signed_sources_requalify_after_each_switch():
    initial = ElectrostaticProblem(conductors=(sphere("T", 2, 9, (0, 0, 0)),
        sphere("P", 1, -3, (100, 0, 0)), sphere("N", 0.5, 2, (0, -200, 0))),
        permittivity_f_per_m=5)
    grounded = switch(initial, "T")
    assert grounded.conductors[0].constraint.charge_c == 0.04
    assert grounded.permittivity_f_per_m == 5
    assert switch(grounded, "T") == grounded


def test_symmetric_irrational_distances_cancel_before_rounding():
    centre = sphere("target", 1, 1, (0,0,0))
    sources = (sphere("plus", 1, 3, (100,100,0)), sphere("minus", 1, -6, (-200,-200,0)))
    after = switch(ElectrostaticProblem(conductors=(centre, *sources)), "target")
    assert after.conductors[0].constraint.charge_c == 0


def test_unrepresentable_nonzero_induced_charge_is_refused():
    initial = ElectrostaticProblem(conductors=(sphere("T", 1, 0, (0,0,0)),
        sphere("source", 1, 5e-324, (100,0,0))))
    with pytest.raises(UnsupportedProblem, match="numerical precision"):
        switch(initial, "T")


def test_exact_geometric_threshold_even_when_hypot_rounds_to_twenty():
    target = sphere("A", 1, 0, (0,0,0))
    def arrangement(y):
        return ElectrostaticProblem(conductors=(target, sphere("B", 1, 1, (12,y,0))))
    with pytest.raises(UnsupportedProblem, match="separation"):
        switch(arrangement(nextafter(16., 0.)), "A")
    assert switch(arrangement(16.), "A").conductors[0].constraint.charge_c == -0.05


def test_refuses_close_or_simultaneous_grounding():
    with pytest.raises(UnsupportedProblem, match="separation"):
        switch(ElectrostaticProblem(conductors=(sphere("A", 1, 0, (0,0,0)), sphere("B", 1, 1, (4,0,0)))), "A")
    problem = ElectrostaticProblem(conductors=(sphere("A", 1, 0, (0,0,0)), sphere("B", 1, 1, (100,0,0))))
    both = ConnectionPlan(problem, (ElectricalLink(("A","B"), "grounded"),))
    with pytest.raises(UnsupportedProblem, match="exactly one"):
        ground_and_disconnect_distant_sphere(both, approximation="centre-monopole")
