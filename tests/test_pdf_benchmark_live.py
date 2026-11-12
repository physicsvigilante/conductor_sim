"""M6 PDF cases: run the *real* FEMM 5 solver, not a mocked adapter.

The pinned FEMM GitHub Actions workflow executes this module with FEMM5_BINDING.
"""
from os import environ

import pytest

from conductor_sim import Femm5Backend, NestedShellBackend
from conductor_sim.pdf_benchmark import load_case

pytestmark = [pytest.mark.femm5,
              pytest.mark.skipif(not environ.get('FEMM5_BINDING'),reason='FEMM 5 not installed')]


@pytest.mark.parametrize('case_id',["p26-conductor-potential", "p29-outer-charge-change",
                                    "p62-ground-outer", "p63-ground-inner", "p69-cavity-central-vs-shifted"])
def test_pdf_case_with_actual_femm5_and_independent_reference(case_id):
    case=load_case(case_id)
    authored=case.author()
    exact=NestedShellBackend()
    numeric=Femm5Backend()
    for label,p in authored.states:
        expected=exact.solve(p)
        with numeric.solve(p) as got:
            assert len(got.conductor_results)==len(expected.conductor_results)
            for observed,reference in zip(got.conductor_results, expected.conductor_results):
                assert observed.name==reference.name
                assert observed.potential_v==pytest.approx(reference.potential_v,rel=.10,abs=.5)
                assert observed.net_charge_c==pytest.approx(reference.net_charge_c,rel=.07,abs=3e-11)
                assert observed.inner_surface_charge_c==pytest.approx(reference.inner_surface_charge_c,rel=.07,abs=3e-11)
            for point in [(0.0,0.0,.06),(.16,0.0,0.0),(.27,0.0,0.0)]:
                # Some benchmark shells may put (.16,0,0) inside metal;
                # potential is still defined there, unlike E on the face.
                assert got.potential_at(point)==pytest.approx(expected.potential_at(point),rel=.15,abs=1.0)
