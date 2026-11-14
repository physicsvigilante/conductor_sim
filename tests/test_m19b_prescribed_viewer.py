"""M19b: stored true fixed-source time playback, independent source checks.

An immutable field never reacts to the moving test charge. A profile is
physical only along the certified ring/disc axis or at the exact time-sampled
particle positions; we must never invent an off-axis potential map.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from math import hypot, pi
from pathlib import Path
import subprocess
import sys

import pytest

from conductor_sim import (
    FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec,
    UniformRingAxisField, UniformDiskAxisField, UniformElectricField,
    UniformSphereField, UniformShellField, UniformRodField,
    integrate_test_charge, sample_test_charge_viewer,
)
from conductor_sim.viewer import render_html
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.prescribed_viewer_validation import validate_prescribed_viewer
from examples.test_charge_motion import CASES, make_case


def norm(v):
    return hypot(*v)


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def build(case,*,cap=7):
    spec=make_case(case)
    result=integrate_test_charge(spec)
    return spec,result,sample_test_charge_viewer(spec,result=result,visual_frames=cap,
                                                 profile_samples=101,resolution=13)


@pytest.mark.parametrize('case',CASES)
def test_all_eight_real_m19a_cases_preserve_full_trajectory(case):
    spec,result,b=build(case)
    meta=b['prescribed_motion']
    assert meta['schema']=='conductor_sim.test_charge_viewer.v1'
    assert len(b['frames'])==7
    assert len(meta['samples'])==len(result.samples)
    assert meta['frame_sample_indices'][0]==0
    assert meta['frame_sample_indices'][-1]==len(result.samples)-1
    assert meta['graph_sample_count']==len(result.samples)
    assert meta['field_frame_count']==len(b['frames'])
    assert b['schema']=='conductor_sim.visualization.v1'
    assert 'prescribed_motion' in b and all(k not in b for k in ('motion_1d','motion_2d','position_sweep'))
    assert meta['field']['sources']==json.loads(json.dumps([s.as_dict() for s in spec.field.sources]))
    assert meta['particle']['position_m']==list(spec.particle.position_m)
    assert meta['particle']['charge_c']==spec.particle.charge_c
    assert 'no back-reaction' in meta['mode']
    assert 'full qV' not in meta['energy_convention'] or 'qV' in meta['energy_convention']
    assert meta['profile_mode']==('axis' if spec.axis is not None else 'elapsed-time')
    assert validate_prescribed_viewer(b)
    for f,i in zip(b['frames'],meta['frame_sample_indices']):
        assert f['motion_sample_index']==i
        assert len(f['charges'])==1
        assert f['charges'][0]['position_m']==pytest.approx(result.samples[i].position_m)
        assert f['charges'][0]['motion_role']=='test particle'
        assert f['conductor_results']==f['geometry']==f['contours']==f['field_lines']==[]
        assert set(v for row in f['grid']['potential_v'] for v in row)=={None}
        assert set(v for row in f['grid']['electric_field_v_per_m'] for v in row)=={None}
        assert f['plane']==b['frames'][0]['plane']
    html=render_html(b)
    assert 'test-charge motion in a fixed electric field' in html.lower()
    assert 'drawPrescribedSources' in html
    assert 'drawAxialChart' in html
    assert json.loads(json.dumps(b,allow_nan=False))==b


def test_axis_only_profile_matches_ring_reference_formula_and_centre_crossing():
    spec,result,b=build('ring-crossing')
    profile=b['prescribed_motion']['profile_samples']
    assert len(profile)==101
    centre=profile[50]
    assert centre['coordinate_m']==pytest.approx(0)
    assert centre['field_parallel_v_m']==pytest.approx(0,abs=1e-15)
    k=1/(4*pi*spec.field.permittivity_f_m)
    ring=spec.field.sources[0]
    assert centre['potential_v']==pytest.approx(k*ring.charge_c/ring.radius_m,rel=1e-12)
    assert profile[35]['field_parallel_v_m']==pytest.approx(-profile[65]['field_parallel_v_m'],rel=1e-12)
    assert profile[35]['potential_v']==pytest.approx(profile[65]['potential_v'],rel=1e-12)
    assert all(p['potential_v'] is not None and p['field_parallel_v_m'] is not None for p in profile)
    assert result.samples[0].potential_energy_j==pytest.approx(
        spec.particle.charge_c*spec.field.potential_at(spec.particle.position_m))


def test_disc_field_has_real_centre_gap_but_potential_continuity():
    spec,_,b=build('disc-approach')
    profile=b['prescribed_motion']['profile_samples']
    assert profile[50]['coordinate_m']==pytest.approx(0,abs=1e-15)
    assert profile[50]['field_parallel_v_m'] is None
    assert profile[50]['potential_v'] is not None
    assert profile[49]['field_parallel_v_m']<0<profile[51]['field_parallel_v_m']
    assert profile[49]['potential_v']==pytest.approx(profile[51]['potential_v'],rel=1e-12)
    for frame in b['frames']:
        assert not any(value is not None for row in frame['grid']['potential_v'] for value in row)


def test_immutable_field_observations_and_correct_full_qv():
    spec,result,b=build('uniform-deflection')
    original=spec.field.as_dict()
    sample=b['prescribed_motion']['samples'][3]
    q=spec.particle.charge_c
    assert sample['force_n']==pytest.approx([q*e for e in sample['electric_field_v_m']])
    assert sample['potential_energy_j']==pytest.approx(q*sample['potential_v'])
    assert sample['interaction_energy_j']==sample['potential_energy_j']
    assert sample['total_energy_j']==pytest.approx(sample['kinetic_energy_j']+sample['potential_energy_j'])
    assert spec.field.as_dict()==original
    assert b['prescribed_motion']['graph_sample_count']>b['prescribed_motion']['field_frame_count']


def test_supplied_result_never_reintegrates(monkeypatch):
    spec=make_case('two-rings');r=integrate_test_charge(spec)
    import conductor_sim.prescribed_viewer as viewer
    def forbidden(*_args,**_kwargs):raise AssertionError('M19a integration must not repeat')
    monkeypatch.setattr(viewer,'integrate_test_charge',forbidden)
    b=viewer.sample_test_charge_viewer(spec,result=r,visual_frames=8)
    assert len(b['frames'])==8


def test_classical_source_guard_two_sample_termination_and_no_exact_impact():
    spec=TestChargeMotionSpec(PrescribedField((UniformDiskAxisField(1e-12,.1),)),
        PointCharge(-1e-9,(0,0,.00012)),1e-6,(0,0,-.3),.005,.005,
        length_scale_m=.1,source_guard_m=.0001)
    r=integrate_test_charge(spec)
    assert len(r.samples)==2 and r.termination=='source-guard'
    b=sample_test_charge_viewer(spec,result=r,profile_samples=101)
    assert len(b['frames'])==2
    assert b['prescribed_motion']['events'][0]['kind']=='source-guard'
    assert 'not an exact collision' in r.events[0].description
    assert b['prescribed_motion']['termination']=='source-guard'


def test_volume_sphere_trajectory_can_cross_transparent_boundary():
    spec,results,b=build('sphere-transit')
    assert any(norm(s.position_m)<.1 for s in results.samples)
    assert any(norm(s.position_m)>.1 for s in results.samples)
    assert b['prescribed_motion']['trajectory_break_indices']==[]
    assert b['prescribed_motion']['termination']=='duration'


def test_rod_source_scene_does_not_claim_a_conducting_rod():
    _,_,b=build('rod-deflection')
    assert b['frames'][0]['conductor_results']==[]
    html=render_html(b)
    assert "kind==='UniformRodField'" in html
    assert 'not a potential heatmap' in html
    assert b['prescribed_motion']['profile_samples'] is None


def test_two_ring_superposition_and_transformed_slice():
    spec,r,b=build('two-rings')
    assert len(b['prescribed_motion']['field']['sources'])==2
    p=r.samples[7].position_m
    assert spec.field.potential_at(p)==pytest.approx(sum(s.potential_at(p) for s in spec.field.sources))
    assert spec.field.electric_field_at(p)==pytest.approx(tuple(sum(s.electric_field_at(p)[i]
        for s in spec.field.sources) for i in range(3)))
    # Physical axis may be tilted/transformed without a fictitious off-axis E.
    ring=UniformRingAxisField(1e-9,.1,center_m=(1.,2.,3.),axis=(1.,0.,0.))
    field=PrescribedField((ring,))
    transformed=TestChargeMotionSpec(field,PointCharge(1e-9,(1.2,2.,3.)),1e-6,
                (-.2,0.,0.),.15,.01,length_scale_m=.1,source_guard_m=1e-4,
                plane_normal=(0.,0.,1.))
    result=integrate_test_charge(transformed)
    view=sample_test_charge_viewer(transformed,result=result,profile_samples=101)
    plane=view['frames'][0]['plane']
    assert plane['v']==pytest.approx((1.,0.,0.))
    assert view['prescribed_motion']['profile_samples'][50]['potential_v']==pytest.approx(
        field.potential_at((1.,2.,3.)))


def test_rejects_wrong_trajectory_and_invalid_frame_cap():
    spec,r,b=build('ring-turning')
    with pytest.raises(ValueError,match='exact authored'):
        sample_test_charge_viewer(replace(spec,mass_kg=2e-6),result=r)
    with pytest.raises(ValueError,match='between 2 and 24'):
        sample_test_charge_viewer(spec,result=r,visual_frames=25)
    with pytest.raises(ValueError,match='odd'):
        sample_test_charge_viewer(spec,result=r,profile_samples=100)
    with pytest.raises(UnsupportedProblem,match='view plane'):
        sample_test_charge_viewer(spec,result=r,extent_m=.001)


@pytest.mark.parametrize('mutation,match',[
    (lambda b: b['prescribed_motion']['samples'][5].update(time_s=0), 'increasing times'),
    (lambda b: b['prescribed_motion']['samples'][5]['position_m'].__setitem__(2,2.), 'saved field/potential|guard|displacement'),
    (lambda b: b['prescribed_motion']['samples'][5].update(potential_v=1000.), 'saved field/potential'),
    (lambda b: b['prescribed_motion']['samples'][5]['force_n'].__setitem__(2,100.), 'F=qE'),
    (lambda b: b['prescribed_motion']['samples'][5].update(potential_energy_j=100.), 'energy must be K\+qV'),
    (lambda b: b['prescribed_motion']['samples'][5].update(interaction_energy_j=100.), 'energy must be K\+qV'),
    (lambda b: b['prescribed_motion']['samples'][5].update(kinetic_energy_j=100.), 'energy must be K\+qV'),
    (lambda b: b['prescribed_motion']['samples'][5].update(energy_drift_j=100.), 'energy must be K\+qV'),
    (lambda b: b['prescribed_motion']['samples'][5]['electric_field_v_m'].__setitem__(2,100.), 'saved field/potential'),
    (lambda b: b['prescribed_motion']['field']['sources'][0].update(charge_c=2e-9), 'saved field/potential'),
    (lambda b: b['prescribed_motion'].update(frame_sample_indices=list(reversed(b['prescribed_motion']['frame_sample_indices']))), 'mapping'),
    (lambda b: b['frames'][1]['charges'][0]['position_m'].__setitem__(2,100.), 'display test particle'),
    (lambda b: b['frames'][1].update(contours=[{}]), 'cannot be shown'),
    (lambda b: b['frames'][1]['grid']['potential_v'][0].__setitem__(0,22.), 'unsolved 2D'),
    (lambda b: b['prescribed_motion']['profile_samples'][0].update(field_parallel_v_m=9e9), 'axial profile'),
])
def test_mutations_rejected_before_unsafe_html(mutation,match):
    _,_,good=build('ring-turning')
    bad=deepcopy(good)
    mutation(bad)
    with pytest.raises(ValueError,match=match):render_html(bad)


def test_field_jump_cannot_be_drawn_as_continuous_axial_field():
    _,_,b=build('disc-approach')
    assert b['prescribed_motion']['profile_samples'][50]['field_parallel_v_m'] is None
    bad=deepcopy(b)
    bad['prescribed_motion']['profile_samples'][50]['field_parallel_v_m']=0.
    with pytest.raises(ValueError,match='null gaps|axial profile'):
        render_html(bad)


def test_suspect_nonaxial_field_map_is_rejected():
    _,_,b=build('uniform-deflection')
    bad=deepcopy(b)
    bad['frames'][0]['field_lines']=[{'points_m':[[0,0],[1,1]]}]
    with pytest.raises(ValueError,match='cannot be shown'):
        render_html(bad)
    bad=deepcopy(b)
    bad['prescribed_motion']['profile_mode']='axis'
    with pytest.raises(ValueError,match='axis-only'):
        render_html(bad)


@pytest.mark.parametrize('case',CASES)
def test_cli_generated_html_embeds_identical_json_and_all_real_samples(case,tmp_path):
    out=tmp_path/(case+'.html')
    result=subprocess.run([sys.executable,'examples/test_charge_viewer.py',
        '--case',case,'--visual-frames','6','--profile-samples','101',
        '--resolution','13','--output',str(out)],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stderr+result.stdout
    assert out.exists() and out.with_suffix('.json').exists()
    data=json.loads(out.with_suffix('.json').read_text())
    html=out.read_text()
    embedded=html.split('<script id="simulation-data" type="application/json">',1)[1].split('</script>',1)[0]
    assert json.loads(embedded)==data
    assert data['prescribed_motion']['example']==case
    assert len(data['frames'])==6
    assert data['prescribed_motion']['graph_sample_count']==len(data['prescribed_motion']['samples'])


def test_preserves_previous_m16_m17_m18_render_modes():
    from conductor_sim import (
        ConductingPlane, ElectrostaticProblem, GroundedPlaneBackend,
        Motion2DSpec, integrate_motion_2d, sample_motion_2d,
    )
    # Older conductor viewer is still a different schema, with real
    # conductor-induced fields, not a prescribed-source test field.
    spec=Motion2DSpec(ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.1)),),
               (ConductingPlane(),)),0,1e-6,(.1,0.,-.02),.025,.005)
    backend=GroundedPlaneBackend()
    physics=integrate_motion_2d(spec,backend)
    old=sample_motion_2d(spec,backend,result=physics,resolution=9,
                         line_steps=10,seeds=[],profile_samples=9,visual_frames=5)
    html=render_html(old)
    assert 'const prescribed=data.prescribed_motion||null' in html
    assert 'motion_2d' in old and 'prescribed_motion' not in old
