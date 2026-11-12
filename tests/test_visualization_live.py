"""Actual FEMM 5 sampling/export regressions; never replace solver outputs."""
from os import environ
from math import isfinite
import pytest
from conductor_sim import (ElectrostaticProblem,PointCharge,SphericalShell,Grounded,
                           FixedNetCharge,Femm5Backend,NestedShellBackend)
from conductor_sim.visualization import SliceSpec,sample_frame,sample_sweep,write_visualization
from conductor_sim.visual_adapters import context_for_solution

pytestmark=[pytest.mark.femm5,pytest.mark.skipif(not environ.get('FEMM5_BINDING'),reason='real FEMM 5 not configured')]


def problem():
    return ElectrostaticProblem((PointCharge(1e-9,(0,0,.035)),PointCharge(.45e-9,(0,0,-.025))),
        (SphericalShell(.1,.12,FixedNetCharge(0),name='inner'),SphericalShell(.2,.22,Grounded(),name='outer')))


def test_real_femm_visual_frame_samples_same_live_solution(tmp_path):
    p=problem()
    spec=SliceSpec.meridian(p,resolution=15,extent_m=.3)
    with Femm5Backend().solve(p) as numerical:
        context=context_for_solution(p,numerical)
        frame=sample_frame(context,spec,seeds=[((.05,0),1)],line_steps=200,profile_samples=9)
        assert frame['backend']=='Femm5Solution'
        assert all(m['kind']=='numerical-core' for m in frame['source_masks'])
        assert len(frame['source_masks'])==2
        assert len(frame['profiles'])==4
        assert 'FEMM' in frame['density_method']
        for j,y in enumerate(frame['grid']['y_m']):
            for i,x in enumerate(frame['grid']['x_m']):
                v=frame['grid']['potential_v'][j][i]
                if v is not None:
                    assert v==pytest.approx(numerical.potential_at(spec.world((x,y))),rel=1e-12,abs=1e-12)
        for curve in frame['profiles']:
            if curve['conductor']=='outer' and curve['surface']=='outer':
                assert max(abs(v) for v in curve['sigma_c_per_m2'])<5e-10
        assert frame['field_lines']
        assert frame['field_lines'][0]['termination']=='surface'


def test_real_femm_sweep_export_closes_sessions(tmp_path):
    backend=Femm5Backend();solutions=[]
    class Recorder:
        def solve(self,p):
            sol=backend.solve(p);solutions.append(sol);return sol
    p=problem();spec=SliceSpec.meridian(p,resolution=9,extent_m=.3)
    bundle=sample_sweep([('state one',p),('state two',p)],Recorder(),plane=spec,seeds=[])
    assert len(solutions)==2 and all(s._closed for s in solutions)
    assert all(f['backend']=='Femm5Solution' for f in bundle['frames'])
    assert write_visualization(bundle,tmp_path/'femm.html').is_file()


def test_real_femm_individual_signed_source_charge_not_just_sum():
    # M4's original tests checked the summed source charge. A zero-net dipole
    # must retain both distinct cores and nonzero cavity fields.
    p=ElectrostaticProblem((PointCharge(1e-9,(0,0,.03)),PointCharge(-1e-9,(0,0,-.03))),
                          (SphericalShell(.1,.12,FixedNetCharge(0)),))
    exact=NestedShellBackend().solve(p)
    with Femm5Backend().solve(p) as numerical:
        assert len(numerical.numerical_sources)==2
        for source in numerical.numerical_sources:
            q=next(c.charge_c for c in p.charges if c.position_m==source.position_m)
            assert source.measured_charge_c==pytest.approx(q,rel=.04,abs=1e-12)
        assert numerical.numerical_source_charge_c==pytest.approx(0,abs=1e-11)
        assert numerical.electric_field_at((.05,0,0))==pytest.approx(exact.electric_field_at((.05,0,0)),rel=.25,abs=20)
        assert numerical.potential_at((.3,0,0))==pytest.approx(0,abs=.1)
