"""M17b: curved playback is a projection of real M17a numerical data."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from math import hypot
from pathlib import Path
import subprocess
import sys

import pytest

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, ExternalSphereBackend, Grounded,
    GroundedPlaneBackend, Motion2DSpec, Motion1DSpec, ParallelPlateBackend,
    PointCharge, SolidSphere, SphericalShell, MultiImageShellBackend,
    integrate_motion_2d, integrate_motion_1d, sample_motion_1d, sample_motion_2d,
)
from conductor_sim.motion_viewer_2d import motion_slice_2d
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.viewer import render_html
from conductor_sim.visualization import SliceSpec


def plane_spec(n=9, *, tangential=.2):
    p=ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.10)),),(ConductingPlane(),))
    return Motion2DSpec(p,0,1e-6,(tangential,0.,-.03),.04,.04/(n-1))


def quick(spec,backend,result=None,*,visual_frames=24,plane=None):
    return sample_motion_2d(spec,backend,result=result,plane=plane,visual_frames=visual_frames,
                            resolution=9,line_steps=10,seeds=[],profile_samples=9)


def test_real_angled_motion_full_samples_and_matching_field_frames():
    spec=plane_spec(51)
    backend=GroundedPlaneBackend()
    result=integrate_motion_2d(spec,backend)
    bundle=quick(spec,backend,result,visual_frames=10)
    meta=bundle['motion_2d']
    assert bundle['schema']=='conductor_sim.visualization.v1'
    assert meta['mode']=='elapsed-time planar classical vector dynamics'
    assert meta['schema']=='conductor_sim.motion_viewer_2d.v1'
    assert meta['graph_sample_count']==len(meta['samples'])==51
    assert meta['field_frame_count']==len(bundle['frames'])==10
    assert meta['frame_sample_indices'][0]==0
    assert meta['frame_sample_indices'][-1]==50
    assert 'position_sweep' not in bundle and 'motion_1d' not in bundle
    assert all(a<b for a,b in zip(meta['frame_sample_indices'],meta['frame_sample_indices'][1:]))
    assert [bundle['frames'][i]['motion_sample_index'] for i in range(10)]==meta['frame_sample_indices']
    for f,idx in zip(bundle['frames'],meta['frame_sample_indices']):
        sample=meta['samples'][idx]
        assert f['charges'][0]['position_m']==pytest.approx(sample['position_m'])
        assert f['charges'][0]['slice_position_m'][2]==pytest.approx(0.,abs=1e-12)
        assert f['backend']=='GroundedPlaneSolution'
        assert all(len(sample[key])==3 for key in (
            'position_m','displacement_m','velocity_m_s','acceleration_m_s2','force_n'))
    assert all(f['plane']==bundle['frames'][0]['plane'] for f in bundle['frames'])
    assert spec.problem.charges[0].position_m==(0.,0.,.1)
    html=render_html(bundle)
    assert 'id="trajectory-toggle"' in html
    assert 'id="motion-vector-legend"' in html
    assert 'drawTrajectory' in html
    assert json.loads(json.dumps(bundle,allow_nan=False))['motion_2d']==meta


def test_stored_result_skips_second_integration_and_solves_exact_frame_count():
    class Counter:
        def __init__(self):self.count=0
        def supports(self,p):return True
        def solve(self,p):
            self.count+=1
            return GroundedPlaneBackend().solve(p)
    backend=Counter()
    spec=plane_spec(20)
    result=integrate_motion_2d(spec,backend)
    baseline=backend.count
    b=quick(spec,backend,result,visual_frames=6)
    assert backend.count-baseline==len(b['frames'])==6
    assert backend.count>baseline
    assert b['motion_2d']['graph_sample_count']==20


def test_plane_anchor_does_not_zoom_out_real_curve():
    conductor=ConductingPlane((10000.,10000.,0.),(0.,0.,1.),Grounded(),'shifted')
    spec=plane_spec()
    spec=replace(spec,problem=replace(spec.problem,conductors=(conductor,)))
    r=integrate_motion_2d(spec,GroundedPlaneBackend())
    sl=motion_slice_2d(spec,r,resolution=9)
    assert sl.x_range_m[1]<1.
    assert sl.y_range_m[1]<1.
    assert all(sl.contains(sl.project(row.position_m)[:2]) for row in r.samples)


def test_spherical_exterior_path_and_cavity_use_fixed_meridian():
    sphere=ElectrostaticProblem((PointCharge(1e-9,(.18,0.,0.)),),
                                 (SolidSphere(.07,Grounded()),))
    cavity=ElectrostaticProblem((PointCharge(1e-9,(.03,0.,.01)),),
                                 (SphericalShell(.10,.12,Grounded()),))
    for spec,backend in [
        (Motion2DSpec(sphere,0,1e-6,(.015,.13,.07),.04,.005),ExternalSphereBackend()),
        (Motion2DSpec(cavity,0,1e-6,(0.,.09,.03),.04,.005),MultiImageShellBackend()),
    ]:
        r=integrate_motion_2d(spec,backend)
        b=quick(spec,backend,r,visual_frames=5)
        assert len(b['frames'])==5
        assert b['motion_2d']['samples'][-1]['position_m']!=b['motion_2d']['samples'][0]['position_m']
        extent=max(c.outer_radius_m for c in spec.problem.conductors)
        sl=b['frames'][0]['plane']
        assert sl['x_range_m'][1]>extent and sl['y_range_m'][1]>extent
        assert all(abs(frame['charges'][0]['slice_position_m'][2])<1e-10 for frame in b['frames'])
        assert 'id="motion-panel"' in render_html(b)


def test_stationary_cavity_center_and_pre_contact_guard():
    cavity=ElectrostaticProblem((PointCharge(1e-9,(0.,0.,0.)),),
                                 (SphericalShell(.1,.12,Grounded()),))
    centered=Motion2DSpec(cavity,0,1e-6,(0.,0.,0.),.04,.005)
    b=quick(centered,MultiImageShellBackend())
    assert all(h['position_m']==[0.,0.,0.] for h in b['motion_2d']['samples'])
    assert all(h['velocity_m_s']==[0.,0.,0.] for h in b['motion_2d']['samples'])
    assert 'id="trajectory"' in render_html(b)
    p=ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.02)),),(ConductingPlane(),))
    guarded=Motion2DSpec(p,0,1e-6,(.02,0.,-.5),.15,.01,surface_guard_m=1e-4)
    r=integrate_motion_2d(guarded,GroundedPlaneBackend())
    assert r.termination=='surface-guard'
    b=quick(guarded,GroundedPlaneBackend(),r,visual_frames=6)
    assert b['motion_2d']['termination']=='surface-guard'
    assert b['motion_2d']['events'][-1]['kind']=='surface-guard'
    assert b['frames'][-1]['charges'][0]['position_m']==pytest.approx(r.samples[-1].position_m)
    assert all(row['position_m'][2]>1e-4 for row in b['motion_2d']['samples'])


def test_rejects_wrong_result_wrong_plane_and_frame_budget():
    spec=plane_spec()
    backend=GroundedPlaneBackend()
    result=integrate_motion_2d(spec,backend)
    with pytest.raises(ValueError,match='exact specification'):
        quick(replace(spec,mass_kg=2e-6),backend,result)
    with pytest.raises(UnsupportedProblem,match='invariant'):
        quick(spec,backend,result,plane=SliceSpec(u=(1,0,0),v=(0,1,0),resolution=9))
    with pytest.raises(UnsupportedProblem,match='trajectory'):
        quick(spec,backend,result,plane=SliceSpec(origin_m=(0.,3.,0.),resolution=9))
    with pytest.raises(ValueError,match='between 2 and 24'):
        quick(spec,backend,result,visual_frames=25)


def test_corrupt_times_forces_slice_and_mixed_modes_fail_closed():
    b=quick(plane_spec(),GroundedPlaneBackend())
    corrupted=deepcopy(b);corrupted['motion_2d']['samples'][1]['time_s']=0
    with pytest.raises(ValueError,match='increasing'):
        render_html(corrupted)
    corrupted=deepcopy(b);corrupted['motion_2d']['samples'][1]['force_n']=[1]
    with pytest.raises(ValueError,match='finite'):
        render_html(corrupted)
    corrupted=deepcopy(b);corrupted['frames'][1]['charges'][0]['position_m'][0]+=.03
    with pytest.raises(ValueError,match='match'):
        render_html(corrupted)
    corrupted=deepcopy(b);corrupted['frames'][1]['motion_sample_index']=0
    with pytest.raises(ValueError,match='match'):
        render_html(corrupted)
    corrupted=deepcopy(b);corrupted['frames'][1]['plane']['u']=[0.,0.,1.]
    with pytest.raises(ValueError,match='slice'):
        render_html(corrupted)
    corrupted=deepcopy(b);corrupted['motion_1d']={}
    with pytest.raises(ValueError,match='mix'):
        render_html(corrupted)
    corrupted=deepcopy(b);corrupted['motion_2d']['samples'][1]['position_m'][0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        render_html(corrupted)


def test_refuse_missing_femm_force_and_clean_up_solution():
    spec=plane_spec()
    trajectory=integrate_motion_2d(spec,GroundedPlaneBackend())
    closed=[]
    class NoForce:
        conductor_results=()
        def close(self):closed.append(True)
    class Numerical:
        def supports(self,p):return True
        def solve(self,p):return NoForce()
    with pytest.raises(UnsupportedProblem):
        quick(spec,Numerical(),trajectory)
    assert closed==[True]


def test_refuse_mismatched_recomputed_frame_force():
    spec=plane_spec()
    result=integrate_motion_2d(spec,GroundedPlaneBackend())
    # Force consistency is checked for every selected frame, not inferred
    # merely from equal-looking point-charge positions.
    wrong=replace(result,samples=(replace(result.samples[0],
                     force_n=(999.,0.,0.)),*result.samples[1:]))
    with pytest.raises(UnsupportedProblem,match='disagrees'):
        quick(spec,GroundedPlaneBackend(),wrong)


def test_cli_standalone_json_has_exact_html_physics(tmp_path):
    path=tmp_path/'m17b.html'
    command=[sys.executable,'examples/particle_motion_2d_viewer.py',
             '--case','plane','--resolution','9','--line-steps','8',
             '--visual-frames','6','--output',str(path)]
    result=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],
                          capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr+result.stdout
    assert path.exists() and path.with_suffix('.json').exists()
    json_data=json.loads(path.with_suffix('.json').read_text())
    html=path.read_text()
    assert json_data['motion_2d']['example']=='plane'
    assert len(json_data['motion_2d']['frame_sample_indices'])==len(json_data['frames'])==6
    embedded=html.split('<script id="simulation-data" type="application/json">',1)[1].split('</script>',1)[0]
    assert json.loads(embedded)==json_data


def test_m16b_still_has_original_scalar_payload_and_no_new_path_ui():
    p=ElectrostaticProblem((PointCharge(1e-9,(0,0,.10)),),(ConductingPlane(),))
    spec=Motion1DSpec(p,0,1e-6,(0,0,1),-.02,.04,.01)
    backend=GroundedPlaneBackend()
    r=integrate_motion_1d(spec,backend)
    b=sample_motion_1d(spec,backend,result=r,resolution=9,line_steps=10,
                       seeds=[],profile_samples=9,visual_frames=5)
    assert 'motion_1d' in b and 'motion_2d' not in b
    assert b['motion_1d']['schema']=='conductor_sim.motion_viewer.v1'
    assert 'const motion1=data.motion_1d||null' in render_html(b)
