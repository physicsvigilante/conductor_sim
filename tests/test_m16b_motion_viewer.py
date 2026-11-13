"""M16b elapsed-time viewer is backed by M16a physics, not a fake position sweep."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from math import isclose
from pathlib import Path
import subprocess
import sys

import pytest

from conductor_sim import (ConductingPlane, ElectrostaticProblem, FixedPotential, Grounded,
                           GroundedPlaneBackend, MultiImageShellBackend, ParallelPlateBackend,
                           PointCharge, SphericalShell, Motion1DSpec, integrate_motion_1d)
from conductor_sim.motion_viewer import sample_motion_1d, motion_slice
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.viewer import render_html
from conductor_sim.visualization import SliceSpec


def plane_spec(n=6):
    p=ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.10)),),(ConductingPlane(name='plane'),))
    return Motion1DSpec(p,0,1e-6,(0.,0.,1.),-.02,.05,.05/(n-1))


def quick(spec, backend, result=None, *, visual_frames=24):
    return sample_motion_1d(spec,backend,result=result,resolution=9,line_steps=10,
                            seeds=[],profile_samples=9,visual_frames=visual_frames)


def test_frame_sources_fields_and_full_timeline_are_same_real_trajectory():
    spec=plane_spec()
    backend=GroundedPlaneBackend()
    original=integrate_motion_1d(spec,backend)
    bundle=quick(spec,backend,original)
    motion=bundle['motion_1d']
    assert bundle['schema']=='conductor_sim.visualization.v1'
    assert bundle['playback'].startswith('Actual elapsed-time trajectory')
    assert motion['mode']=='elapsed-time classical dynamics'
    assert motion['schema']=='conductor_sim.motion_viewer.v1'
    assert len(motion['samples'])==len(bundle['frames'])==6
    assert motion['frame_sample_indices']==list(range(6))
    assert 'position_sweep' not in bundle
    assert motion['max_abs_energy_drift_j']==original.max_abs_energy_drift_j
    for frame,row in zip(bundle['frames'],motion['samples']):
        assert frame['charges'][0]['position_m']==pytest.approx(row['position_m'])
        assert frame['backend']=='GroundedPlaneSolution'
        assert frame['motion_sample_index']==motion['samples'].index(row)
        assert row['kinetic_energy_j']==pytest.approx(.5*spec.mass_kg*row['velocity_m_s']**2)
        assert row['total_energy_j']==pytest.approx(row['kinetic_energy_j']+row['interaction_energy_j'])
        assert row['time_s']>=0
    assert 'id="motion-panel"' in render_html(bundle)
    assert json.loads(json.dumps(bundle,allow_nan=False))['motion_1d']['termination']=='duration'
    assert spec.problem.charges[0].position_m==(0.,0.,.1)


def test_m16b_does_not_reintegrate_m16a_result_in_viewer_sampling():
    class Counted:
        def __init__(self):self.solves=0
        def supports(self, p):return True
        def solve(self,p):
            self.solves+=1
            return GroundedPlaneBackend().solve(p)
    spec=plane_spec(8)
    backend=Counted()
    trajectory=integrate_motion_1d(spec,backend)
    old_count=backend.solves
    bundle=quick(spec,backend,trajectory)
    assert backend.solves-old_count==len(bundle['frames'])==8
    assert backend.solves > old_count


def test_trajectory_larger_than_24_keeps_full_numerical_graph_and_guard_events():
    spec=plane_spec(51)
    backend=GroundedPlaneBackend()
    r=integrate_motion_1d(spec,backend)
    bundle=quick(spec,backend,r,visual_frames=10)
    motion=bundle['motion_1d']
    assert len(motion['samples'])==51 and len(bundle['frames'])==10
    assert motion['graph_sample_count']==51
    assert motion['frame_sample_indices'][0]==0
    assert motion['frame_sample_indices'][-1]==50
    assert all(a<b for a,b in zip(motion['frame_sample_indices'],motion['frame_sample_indices'][1:]))
    assert max(b-a for a,b in zip(motion['frame_sample_indices'],motion['frame_sample_indices'][1:]))<=8
    assert all(frame['motion_sample_index']==index for frame,index in zip(bundle['frames'],motion['frame_sample_indices']))
    assert motion['events']==[e for e in r.as_dict()['events']]


def test_guarded_approach_not_claimed_impact_and_last_frame_is_terminal():
    p=ElectrostaticProblem((PointCharge(1e-9,(0,0,.02)),),(ConductingPlane(),))
    spec=Motion1DSpec(p,0,1e-6,(0,0,1),-.5,.15,.01,surface_guard_m=1e-4)
    result=integrate_motion_1d(spec,GroundedPlaneBackend())
    assert result.termination=='surface-guard'
    bundle=quick(spec,GroundedPlaneBackend(),result,visual_frames=6)
    rows=bundle['motion_1d']['samples']
    assert rows[-1]['position_m'][2]>spec.surface_guard_m
    assert bundle['frames'][-1]['charges'][0]['position_m']==pytest.approx(rows[-1]['position_m'])
    assert any(event['kind']=='surface-guard' for event in bundle['motion_1d']['events'])
    assert 'exact' in render_html(bundle).lower()  # explicitly qualified, never "exact impact" claim


def test_exact_zero_motion_center_has_no_spurious_sample_force_or_arrow():
    p=ElectrostaticProblem((PointCharge(1e-9,(0,0,0)),),(
        SphericalShell(.1,.12,Grounded(),name='cavity'),))
    spec=Motion1DSpec(p,0,1e-6,(0,0,1),0,.04,.01)
    backend=MultiImageShellBackend()
    r=integrate_motion_1d(spec,backend)
    b=quick(spec,backend,r)
    assert all(abs(row['velocity_m_s'])<1e-15 and abs(row['force_n'])<1e-17 for row in b['motion_1d']['samples'])
    assert all(frame['charges'][0]['position_m']==[0.,0.,0.] for frame in b['frames'])
    assert b['frames'][0]['geometry'][0]['inner_radius_m']==pytest.approx(.1)


def test_turned_motion_preserves_signed_velocity_and_nearest_turning_sample():
    spec=replace(plane_spec(),initial_velocity_m_s=.03,duration_s=.25,sample_interval_s=.005)
    backend=GroundedPlaneBackend()
    result=integrate_motion_1d(spec,backend)
    assert any(e.kind=='turning-point' for e in result.events)
    b=quick(spec,backend,result,visual_frames=8)
    assert len(b['frames'])==8 and len(b['motion_1d']['samples'])>8
    assert any(s['velocity_m_s']>0 for s in b['motion_1d']['samples'])
    assert any(s['velocity_m_s']<0 for s in b['motion_1d']['samples'])
    turning=result.events[0].time_s
    nearest=min(range(1,len(result.samples)-1),key=lambda i:abs(result.samples[i].time_s-turning))
    assert nearest in b['motion_1d']['frame_sample_indices']


def test_custom_oblique_slice_and_wrong_trajectory_are_rejected():
    spec=plane_spec()
    backend=GroundedPlaneBackend()
    result=integrate_motion_1d(spec,backend)
    with pytest.raises(ValueError,match='exact specification'):
        quick(replace(spec,mass_kg=2e-6),backend,result)
    with pytest.raises(UnsupportedProblem,match='not entirely'):
        sample_motion_1d(spec,backend,result=result,plane=SliceSpec(origin_m=(0,3,0)),
                         resolution=9,seeds=[],line_steps=10,profile_samples=9)
    with pytest.raises(ValueError,match='between 2 and 24'):
        quick(spec,backend,result,visual_frames=25)
    with pytest.raises(ValueError,match='between 2 and 24'):
        quick(spec,backend,result,visual_frames=1)


def test_arbitrary_plane_tangential_anchor_cannot_blow_up_default_view():
    c=ConductingPlane((100000,0,0),(0,0,1),Grounded(),'offset')
    p=ElectrostaticProblem((PointCharge(1e-9,(0,0,.10)),),(c,))
    spec=Motion1DSpec(p,0,1e-6,(0,0,1),.02,.03,.01)
    result=integrate_motion_1d(spec,GroundedPlaneBackend())
    sl=motion_slice(spec,result,resolution=9)
    assert sl.x_range_m[1]<.5 and sl.y_range_m[1]<.5
    assert sl.contains(sl.project(result.samples[-1].position_m)[:2])
    assert abs(sl.project(p.charges[0].position_m)[2])<1e-11


def test_biased_plates_electron_is_actual_elapsed_nanosecond_motion():
    lower=ConductingPlane(name='lower')
    upper=ConductingPlane((0,0,.2),(0,0,-1),FixedPotential(20.),'upper')
    charge=-1.602176634e-19
    mass=9.1093837139e-31
    p=ElectrostaticProblem((PointCharge(charge,(0,0,.1)),),(lower,upper))
    spec=Motion1DSpec(p,0,mass,(0,0,1),0.,5e-8,5e-9)
    backend=ParallelPlateBackend(image_order=48)
    result=integrate_motion_1d(spec,backend)
    bundle=quick(spec,backend,result)
    assert len(bundle['frames'])==11
    assert bundle['motion_1d']['units']['time']=='s'
    assert bundle['motion_1d']['samples'][-1]['time_s']==pytest.approx(5e-8)
    assert result.samples[-1].velocity_m_s>800000
    assert abs(bundle['motion_1d']['max_abs_energy_drift_j'])<1e-17
    assert bundle['frames'][0]['charges'][0]['charge_c']<0


def test_render_html_fail_closed_on_mismatched_fields_times_and_sweep():
    spec=plane_spec(4);backend=GroundedPlaneBackend();b=quick(spec,backend)
    broken=deepcopy(b);broken['frames'][2]['charges'][0]['position_m'][2]+=.02
    with pytest.raises(ValueError,match='match'):
        render_html(broken)
    broken=deepcopy(b);broken['motion_1d']['samples'][1]['time_s']=0
    with pytest.raises(ValueError,match='increasing'):
        render_html(broken)
    broken=deepcopy(b);broken['motion_1d']['frame_sample_indices']=[0,2,1,3]
    with pytest.raises(ValueError,match='indices'):
        render_html(broken)
    broken=deepcopy(b);broken['position_sweep']={}
    with pytest.raises(ValueError):
        render_html(broken)
    broken=deepcopy(b);broken['motion_1d']['samples'][1]['velocity_m_s']=float('nan')
    with pytest.raises(ValueError,match='finite'):
        render_html(broken)


def test_cli_standalone_html_and_json_match_exactly(tmp_path):
    out=tmp_path/'viewer.html'
    cmd=[sys.executable,'examples/particle_motion_viewer.py','--case','plane',
         '--duration-s','.025','--sample-interval-s','.005',
         '--resolution','9','--line-steps','9','--visual-frames','6',
         '--output',str(out)]
    r=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=50)
    assert r.returncode==0,r.stderr+r.stdout
    assert out.exists() and out.with_suffix('.json').exists()
    data=json.loads(out.with_suffix('.json').read_text())
    html=out.read_text()
    assert data['motion_1d']['example']=='plane'
    assert len(data['frames'])==len(data['motion_1d']['frame_sample_indices'])==6
    assert html.count('id="motion-panel"')==1
    assert json.loads(html.split('<script id="simulation-data" type="application/json">',1)[1].split('</script>',1)[0])==data


def test_refuse_observationless_numerical_backend_and_close_owned_solution():
    spec=plane_spec(3)
    trajectory=integrate_motion_1d(spec,GroundedPlaneBackend())
    closed=[]
    class NoForceSolution:
        conductor_results=()
        def close(self):closed.append(True)
    class Numerical:
        def solve(self,p):return NoForceSolution()
        def supports(self,p):return True
    with pytest.raises(UnsupportedProblem):
        quick(spec,Numerical(),trajectory)
    assert closed==[True]
