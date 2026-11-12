"""Actual FEMM 5 checks for *unseen* M6b authored JEE configurations."""
from math import isclose
from os import environ

import pytest

from conductor_sim import Femm5Backend, NestedShellBackend
from conductor_sim.solver import UnsupportedProblem
from m6b.attempts import SUBMISSIONS
from m6b.evaluate import observe

pytestmark=[pytest.mark.femm5,
    pytest.mark.skipif(not environ.get('FEMM5_BINDING'),reason='real FEMM 5 not configured')]


@pytest.mark.parametrize('case_id',[
    'two-shell-negative-net','grounded-outer-shield',
    'offcentre-negative-grounded','two-positive-separated-sources',
    'translated-rotated-cavity-charge',
])
def test_heldout_authored_models_against_real_femm5(case_id):
    authored=SUBMISSIONS[case_id]
    p=authored.states[0][1]
    reference=NestedShellBackend().solve(p)
    with Femm5Backend().solve(p) as numeric:
        for probe in authored.probes:
            expected=observe(reference,probe)
            got=observe(numeric,probe)
            # These are sampled-FEMM benchmark tolerances, not uniform bounds.
            if probe.kind in ('shell_net_charge','inner_total','outer_total'):
                assert got==pytest.approx(expected,rel=.09,abs=3e-11)
            elif probe.kind=='shell_voltage':
                assert got==pytest.approx(expected,rel=.18,abs=1.3)
            elif probe.kind=='potential':
                assert got==pytest.approx(expected,rel=.18,abs=1.5)
            else:
                assert got==pytest.approx(expected,rel=.35,abs=130)
        if case_id=='two-positive-separated-sources':
            charges=sorted(s.measured_charge_c for s in numeric.numerical_sources)
            assert charges==pytest.approx([.3e-9,.7e-9],rel=.06)
            assert len(numeric.numerical_sources)==2
        if case_id=='translated-rotated-cavity-charge':
            assert numeric.symmetry_axis==pytest.approx((.6,.8,0),abs=1e-12)


def test_heldout_noncollinear_femm_refusal_without_fallback():
    problem=SUBMISSIONS['noncollinear-math-only'].states[0][1]
    assert not Femm5Backend().supports(problem)
    with pytest.raises(UnsupportedProblem):
        Femm5Backend().solve(problem)


def test_heldout_real_femm_two_equilibria_stored_separately():
    s=SUBMISSIONS['grounded-two-state-sweep']
    observed=[]
    for label,p in s.states:
        with Femm5Backend().solve(p) as numeric:
            observed.append(numeric.conductor_by_name('inner').potential_v)
            assert abs(numeric.potential_at((.3,0,0)))<1
    assert observed[0]>0 and observed[1]>0
    assert observed[1]/observed[0]==pytest.approx(3.,rel=.08)
