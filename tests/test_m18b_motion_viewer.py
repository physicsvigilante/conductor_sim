"""Multi-source viewer contract: complete fields, stable identities, no new ODE."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from math import hypot
import pytest
from conductor_sim import (ConductingPlane, ElectrostaticProblem, Grounded,
    GroundedPlaneBackend, Motion2DSpec, PointCharge, SphericalShell,
    MultiImageShellBackend, integrate_motion_2d, sample_motion_2d, motion_slice_2d)
from conductor_sim._motion_validation import planar_path_breaks
from conductor_sim.observables import force_energy as read_force_energy
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.visualization import SliceSpec, write_visualization
from conductor_sim.viewer import render_html
from examples.particle_motion_fixed_charges import compose


def quick(spec,backend,**kwargs):
    return sample_motion_2d(spec,backend,resolution=9,seeds=[],line_steps=8,
                            profile_samples=9,visual_frames=3,**kwargs)


@pytest.mark.parametrize('case',['plane','plates','sphere','cavity','nested','guard'])
def test_all_models_have_matching_fields_complete_source_inventory_and_full_time_series(case):
    spec,backend=compose(case)
    trajectory=integrate_motion_2d(spec,backend)
    bundle=quick(spec,backend,result=trajectory)
    motion=bundle['motion_2d']
    assert len(motion['samples'])==len(trajectory.samples)>len(bundle['frames'])
    assert motion['moving_charge_index']==spec.charge_index
    assert motion['termination']==('charge-guard' if case=='guard' else 'duration')
    for frame,idx in zip(bundle['frames'],motion['frame_sample_indices']):
        by_index={q['charge_index']:q for q in frame['charges']}
        assert set(by_index)==set(range(len(spec.problem.charges)))
        state=spec.problem_at(trajectory.samples[idx].position_m)
        for i,q in enumerate(state.charges):
            displayed=by_index[i]
            assert displayed['position_m']==list(q.position_m)
            assert displayed['charge_c']==q.charge_c
            assert displayed['motion_role']==('moving' if i==spec.charge_index else 'held')
        # Compare a saved field-map value with the ORIGINAL complete backend.
        plane=SliceSpec(**frame['plane']);sol=backend.solve(state)
        observed=False
        for j,row in enumerate(frame['grid']['potential_v']):
            for i,value in enumerate(row):
                if value is not None and frame['grid']['region'][j][i]=='vacuum':
                    point=plane.world((frame['grid']['x_m'][i],frame['grid']['y_m'][j]))
                    assert value==pytest.approx(sol.potential_at(point),rel=1e-12,abs=1e-20)
                    observed=True;break
            if observed:break
        assert observed
    assert len(render_html(bundle))>1000
    assert spec.problem.charges[spec.charge_index].position_m==trajectory.samples[0].position_m
    assert 'position_sweep' not in bundle and 'motion_1d' not in bundle


def test_saved_result_avoids_second_integration_and_matches_original_index_not_sorted_order():
    spec,backend=compose('plane')  # original moving index 1, sorted sample index 0
    result=integrate_motion_2d(spec,backend)
    class Counting:
        def __init__(self):self.seen=[]
        def supports(self,p):return backend.supports(p)
        def solve(self,p):self.seen.append(p);return backend.solve(p)
    counted=Counting();bundle=quick(spec,counted,result=result)
    assert len(counted.seen)==len(bundle['frames'])==3
    for frame,p in zip(bundle['frames'],counted.seen):
        assert p.charges[0]==spec.problem.charges[0]
        assert frame['charges'][0]['motion_role']=='moving'
        assert frame['charges'][0]['charge_index']==1


def test_multiple_equal_signed_sites_are_not_confused_and_reordering_is_safe():
    p=ElectrostaticProblem((PointCharge(1e-10,(.04,0,.1)),
        PointCharge(1e-10,(0,0,.12)),PointCharge(1e-10,(-.06,0,.14))),(ConductingPlane(),))
    s=Motion2DSpec(p,1,1e-6,(.01,0,0),.02,.005)
    b=quick(s,GroundedPlaneBackend())
    for frame in b['frames']:
        assert {q['charge_index'] for q in frame['charges']}=={0,1,2}
        assert sum(q['motion_role']=='held' for q in frame['charges'])==2
    # Loader accepts a different serialization order, not a changed identity.
    changed=deepcopy(b)
    for frame in changed['frames']:frame['charges'].reverse()
    render_html(changed)


def test_fixed_sources_are_inside_view_even_with_far_tangential_plane_anchor():
    s,b=compose('plane')
    p=replace(s.problem,charges=(PointCharge(-1e-10,(2.,0.,.12)),s.selected_charge),
              conductors=(ConductingPlane((1000.,2000.,0.)),))
    s=replace(s,problem=p)
    r=integrate_motion_2d(s,b);plane=motion_slice_2d(s,r,resolution=9)
    assert plane.x_range_m[1]-plane.x_range_m[0]<4
    assert plane.contains(plane.project(s.stationary_charges[0].position_m)[:2])
    with pytest.raises(UnsupportedProblem,match='every held charge'):
        quick(s,b,result=r,plane=SliceSpec(resolution=9))


def test_translated_rotated_and_stationary_center_starts():
    center=(.2,-.3,.4)
    p=ElectrostaticProblem((PointCharge(1e-10,center),
        PointCharge(1e-10,(.24,-.28,.4))),(SphericalShell(.1,.12,Grounded(),center),))
    s=Motion2DSpec(p,0,1e-6,(0,0,0),.02,.005)
    bundle=quick(s,MultiImageShellBackend())
    assert all(abs(q['slice_position_m'][2])<1e-12 for f in bundle['frames'] for q in f['charges'])
    assert bundle['motion_2d']['samples'][-1]['position_m']!=list(center)


@pytest.fixture(scope='module')
def plane_bundle():
    s,b=compose('plane');return quick(s,b)


@pytest.mark.parametrize('bad',['drop','shift','sign','role','duplicate-id','moving-id','projection','fixed-meta','unmapped','mixed'])
def test_corrupt_source_inventory_is_rejected_without_overwriting_file(plane_bundle,tmp_path,bad):
    bundle=deepcopy(plane_bundle)
    sources=bundle['frames'][1]['charges'];fixed=next(q for q in sources if q['motion_role']=='held')
    if bad=='drop':sources.remove(fixed)
    elif bad=='shift':fixed['position_m'][0]+=.01
    elif bad=='sign':fixed['charge_c']*=-1
    elif bad=='role':fixed['motion_role']='moving'
    elif bad=='duplicate-id':fixed['charge_index']=bundle['motion_2d']['moving_charge_index']
    elif bad=='moving-id':bundle['motion_2d']['moving_charge_index']=0
    elif bad=='projection':fixed['slice_position_m']=list(fixed['slice_position_m']);fixed['slice_position_m'][0]+=.1
    elif bad=='fixed-meta':bundle['motion_2d']['stationary_charges'][0]['charge_c']*=-1
    elif bad=='unmapped':del fixed['charge_index']
    elif bad=='mixed':bundle['motion_1d']={}
    out=tmp_path/'kept.html';out.write_text('previous accepted file')
    with pytest.raises(ValueError):write_visualization(bundle,out)
    assert out.read_text()=='previous accepted file'


def test_matching_backend_capability_is_checked_before_acquiring_fields():
    s,b=compose('plane');r=integrate_motion_2d(s,b)
    class Refusing:
        def supports(self,p):return False
        def solve(self,p):raise AssertionError('must not acquire')
    with pytest.raises(UnsupportedProblem,match='backend'):
        quick(s,Refusing(),result=r)


@pytest.mark.parametrize('fault',['no-force','inventory','force','energy'])
def test_bad_field_provider_closes_owned_solution_and_never_falls_back(fault):
    s,b=compose('plane');r=integrate_motion_2d(s,b);closed=[]
    class Broken:
        def __init__(self,p):self.p=p
        def close(self):closed.append(True)
        def force_energy(self):
            if fault=='no-force':raise UnsupportedProblem('no force implementation')
            data=read_force_energy(b.solve(self.p))
            if fault=='inventory':return replace(data,charges=data.charges[:-1])
            if fault=='energy':return replace(data,energy=replace(data.energy,background_j=1.))
            moving=self.p.charges[s.charge_index]
            return replace(data,charges=tuple(replace(q,background_force_n=(1.,0.,0.)) if q.source==moving else q for q in data.charges))
    class Backend:
        def supports(self,p):return True
        def solve(self,p):return Broken(p)
    with pytest.raises(UnsupportedProblem):quick(s,Backend(),result=r)
    assert closed==[True]


def test_sparse_chord_through_held_charge_guard_is_not_drawn(plane_bundle):
    bundle=deepcopy(plane_bundle);motion=bundle['motion_2d']
    fixed=motion['stationary_charges'][0]['position_m']
    motion['samples']=[dict(position_m=[fixed[0]-.03,fixed[1],fixed[2]]),
                       dict(position_m=[fixed[0]+.03,fixed[1],fixed[2]])]
    assert planar_path_breaks(motion,bundle['frames'][0])==[1]


def test_round_trip_html_and_cli(tmp_path):
    target=tmp_path/'held.html'
    proc=subprocess.run([sys.executable,'examples/particle_motion_fixed_viewer.py','--case','plane',
        '--resolution','9','--line-steps','8','--visual-frames','3','--output',str(target)],
        text=True,capture_output=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    meta=json.loads(target.with_suffix('.json').read_text())
    text=target.read_text();embedded=text.split('<script id="simulation-data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(embedded)==meta
    assert 'held-source' in text.lower() and len(meta['frames'])==3
