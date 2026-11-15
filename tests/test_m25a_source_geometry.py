"""M25a: physically honest shared geometry across qualified motion investigations.

No force/field source is modified or sampled by this presentation service. The
reserved transfer is NOT executed in this test module; run it only post-freeze.
"""
from __future__ import annotations

import json
from math import cos, hypot, pi, sin, sqrt
from dataclasses import FrozenInstanceError

import pytest

from conductor_sim import (
    FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec as ChargeMotionSpec,
    UniformDiskAxisField, UniformElectricField, UniformGravity,
    UniformRingAxisField, UniformRodField, UniformShellField,
    UniformSphereField, UnsupportedProblem,
)
from conductor_sim.driven_motion import (
    DrivenModel, DrivenStudy, FieldPhase, LinearDrag, UniformFieldSchedule,
    investigate_driven_motion,
)
from conductor_sim.driven_viewer import render_driven_run
from conductor_sim.motion_event_viewer import render_event_run
from conductor_sim.motion_events import EventPolicy, EventStudy, PlaneEvent, investigate_motion
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.source_geometry import source_geometry
from examples.public_api_event_trial import author
from examples.mirrored_rods import make_case


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _spec(sources, *, position=(.15, 0., 0.), velocity=(0., 0., 0.), normal=(0., 1., 0.), scale=1.):
    return ChargeMotionSpec(PrescribedField(tuple(sources)), PointCharge(-1e-9, position),
        2e-9, velocity, .15, .05, plane_normal=normal, length_scale_m=scale,
        source_guard_m=.001)


def test_world_basis_is_orthonormal_and_source_inventory_is_stable():
    model,_=author('coaxial-rings')
    original=model.as_dict()
    g=source_geometry(model.spec,model.base.source_annotations)
    assert g['schema']=='conductor_sim.source_geometry.v1'
    assert [_['source_id'] for _ in g['entries']]==['ring:A','ring:B']
    assert [e['shape'] for e in g['entries']]==['ring-projection']*2
    assert [e['charge_c'] for e in g['entries']]==[4e-11,-3e-11]
    assert model.as_dict()==original
    assert source_geometry(model.spec,model.base.source_annotations)==g
    u,v,n=g['u'],g['v'],g['normal']
    assert hypot(*u)==pytest.approx(1.,abs=2e-15)
    assert hypot(*v)==pytest.approx(1.,abs=2e-15)
    assert hypot(*n)==pytest.approx(1.,abs=2e-15)
    assert _dot(u,v)==pytest.approx(0.,abs=2e-15)
    assert _dot(v,n)==pytest.approx(0.,abs=2e-15)
    assert _dot(u,n)==pytest.approx(0.,abs=2e-15)


@pytest.mark.parametrize('case',('coaxial-rings','mirror-rod'))
def test_geometry_does_not_change_motion_or_model_revision(case):
    model,q=author(case)
    snap=model.as_dict()
    study=DrivenStudy(model,q,EventPolicy(.01,1e-9,1e-10,1e-9))
    run=investigate_driven_motion(study)
    baseline=json.dumps(run.as_dict(),sort_keys=True)
    first=render_driven_run(run)
    second=render_driven_run(run)
    assert first==second and baseline==json.dumps(run.as_dict(),sort_keys=True)
    assert model.as_dict()==snap
    assert 'source_geometry' in first and 'source-geometry' in first
    assert 'function drawSourceGeometry(' in first
    assert first.count('function drawSourceGeometry(')==1


def test_ring_radius_world_axis_and_full_circle_orthographic_projection():
    model,_=author('coaxial-rings')
    g=source_geometry(model.spec,model.base.source_annotations)
    src=model.spec.field.sources[0]
    e=g['entries'][0];a,b=e['rim_basis_uv'];r=src.radius_m
    assert e['edge_on'] is True
    assert e['status']=='orthographic-projection'
    assert _dot(src.axis,g['normal'])==pytest.approx(0.,abs=1e-14)
    assert abs(a[0]*b[1]-a[1]*b[0])<1e-14
    assert max(hypot(*a),hypot(*b))==pytest.approx(r,abs=1e-14)
    # Independent axial geometry. This ring lies in a true world yz plane.
    for theta in (0., pi/6,pi/2,pi,4*pi/3,2*pi):
        p=(src.center_m[0], r*cos(theta), r*sin(theta))
        delta=tuple(p[i]-g['origin_m'][i] for i in range(3))
        uv=(_dot(delta,g['u']),_dot(delta,g['v']))
        # Parameterization may start at an arbitrary angle, but the projected
        # line must have x fixed and z within +/- physical radius.
        assert uv[0]==pytest.approx(e['center_uv'][0],abs=1e-14)
        assert abs(uv[1]-e['center_uv'][1])<=r+1e-14
        assert abs(uv[1]-e['center_uv'][1])==pytest.approx(abs(r*sin(theta)),abs=1e-14)


def test_ring_source_world_rotation_not_assumed_aligned_to_z():
    r=.2
    model=_spec((UniformRingAxisField(1e-10,r,center_m=(.05,.1,.02),axis=(0.,1.,0.)),),
        position=(.05,.15,.02), velocity=(0.,.01,0.), normal=(1.,0.,0.))
    g=source_geometry(model,(('ring:rot','Rotated ring'),))
    e=g['entries'][0]
    assert e['edge_on']
    assert _dot(e['world_axis'], g['u'])==pytest.approx(1.,abs=1e-14) or _dot(e['world_axis'], g['v'])==pytest.approx(1.,abs=1e-14)
    assert abs(e['signed_normal_offset_m'])<1e-13
    assert max(hypot(*x) for x in e['rim_basis_uv'])==pytest.approx(r)


def test_mirrored_off_plane_points_are_omitted_and_rod_remains():
    model,_=author('mirror-rod')
    g=source_geometry(model.spec,model.base.source_annotations)
    p1,p2,rod=g['entries']
    assert (p1['shape'],p2['shape'],rod['shape'])==('omitted','omitted','rod')
    assert (p1['status'],p2['status'])==('omitted-out-of-plane',)*2
    assert p1['signed_normal_offset_m']==pytest.approx(.18)
    assert p2['signed_normal_offset_m']==pytest.approx(-.18)
    assert p1['bounds_uv']==p2['bounds_uv']==[]
    assert rod['signed_endpoint_offsets_m']==pytest.approx([0,0])
    assert len(rod['bounds_uv'])==2
    assert model.spec.field.electric_field_at(model.spec.particle.position_m) != (0.,0.,0.)


def test_qualified_m22c_rods_are_omitted_not_projected_to_in_plane():
    for case in ('P1','P2'):
        model=make_case(case)
        g=source_geometry(model.spec,model.source_annotations)
        assert g['entries'][0]['shape']=='omitted'
        assert g['entries'][1]['shape']=='omitted'
        assert g['entries'][0]['bounds_uv']==g['entries'][1]['bounds_uv']==[]
        assert any(abs(x)>0 for x in g['entries'][0]['signed_endpoint_offsets_m'])
        assert g['entries'][0]['notes'].endswith('Full 3D field remains active.')
    # The in-plane point in P2 stays physically shown; no extra source invented.
    assert g['entries'][2]['shape']=='point'


def test_offplane_tilted_rods_not_flattened_to_in_plane_chord():
    # Use the qualified, tilted mirror pair: both rods are wholly off the
    # physical plane but their 2D chords would be misleading.
    model=make_case('P2')
    e=source_geometry(model.spec,model.source_annotations)['entries'][0]
    assert e['shape']=='omitted'
    assert e['signed_endpoint_offsets_m'][0]*e['signed_endpoint_offsets_m'][1]>0


def test_normal_aligned_rod_crossing_slice_is_not_drawn_as_in_plane_rod():
    rod=UniformRodField(1e-10,(.25,-.08,0.),(.25,.08,0.))
    spec=_spec((rod,),position=(0.,0.,0.),velocity=(.01,0.,0.))
    entry=source_geometry(spec,(('rod:through','Rod across physical plane'),))['entries'][0]
    assert entry['shape']=='omitted'
    assert entry['signed_endpoint_offsets_m']==pytest.approx([-.08,+.08])
    assert entry['bounds_uv']==[]
    assert 'Full 3D field remains active' in entry['notes']


def test_sphere_shell_slice_uses_true_intersection_not_3d_outline():
    srcs=(UniformSphereField(1e-10,.1,(.35,0.,0.)),
          UniformShellField(-1e-10,.2,(.5,0.,0.)))
    spec=_spec(srcs)
    es=source_geometry(spec,(('sphere:a','Charged insulating volume'),('shell:b','Charged nonconducting shell')))['entries']
    assert [e['shape'] for e in es]==['sphere-slice','shell-slice']
    assert [e['slice_radius_m'] for e in es]==pytest.approx([.1,.2])
    assert 'nonconducting' in es[0]['notes'] and 'nonconducting' in es[1]['notes']


def test_disk_source_draws_disc_not_rod_and_retains_axis_only_qualification():
    d=UniformDiskAxisField(2e-10,.12,center_m=(0.,0.,0.),axis=(1.,0.,0.))
    spec=_spec((d,),position=(.2,0.,0.),velocity=(-.01,0.,0.))
    e=source_geometry(spec,(('disc:axial','Nonconducting disc'),))['entries'][0]
    assert e['shape']=='disk-projection' and e['edge_on']
    assert e['radius_m']==pytest.approx(.12)
    assert 'axis-only' in e['notes']
    with pytest.raises(UnsupportedProblem,match='symmetry axis'):
        d.electric_field_at((.1,.01,0.))


def test_uniform_field_arrow_glyphs_do_not_claim_field_lines():
    spec=_spec((UniformElectricField((100.,0.,-50.)),),position=(.1,0.,0.))
    e=source_geometry(spec,(('E:held','Applied static uniform E'),))['entries'][0]
    assert e['shape']=='field-arrows' and e['status']=='background-field'
    assert e['field_uv']==pytest.approx((100.,-50.))
    assert 'not field lines' in e['notes']


@pytest.mark.parametrize('bad',((('x','A'),('x','B')), (('x','A'),), (('bad/id','Invalid'),('x','B')), (('x','a\ninvalid'),('y','B'))))
def test_malformed_or_mismatched_id_inventory_rejected(bad):
    model,_=author('coaxial-rings')
    with pytest.raises(ValueError):source_geometry(model.spec,bad)


def test_invalid_view_basis_and_unsupported_object_refused():
    spec=_spec(())
    # Tamper only inside test; the saved typed model is immutable normally.
    object.__setattr__(spec,'plane_normal',(0.,0.,0.))
    with pytest.raises(ValueError,match='normal'):source_geometry(spec,())
    with pytest.raises(TypeError):source_geometry(object(),())


def test_physical_refusal_of_ring_with_transverse_scheduled_field_is_retained():
    model,_=author('coaxial-rings')
    with pytest.raises(UnsupportedProblem):
        DrivenModel(model.base,model.drag,
            UniformFieldSchedule((FieldPhase(0.,(0.,0.,0.)),FieldPhase(.2,(0.,10.,0.)))),
            'bad:transverse','Teacher invalid variation: off-axis ring field not qualified.')


def test_m24a_event_viewer_consumes_same_geometry_without_reconstructing_model():
    model,q=author('coaxial-rings')
    study=EventStudy(model.base,(q[1],),EventPolicy(.01,1e-9,1e-10,1e-9))
    run=investigate_motion(study)
    baseline=run.as_dict()
    html=render_event_run(run,title='Shared ring event observation')
    assert 'source_geometry' in html and html.count('function drawSourceGeometry(')==1
    assert 'Physical source geometry' in html
    assert run.as_dict()==baseline
    g=source_geometry(model.spec,model.base.source_annotations)
    assert [e['shape'] for e in g['entries']]==['ring-projection','ring-projection']


def test_geometry_is_identical_for_same_scenario_across_two_investigation_families():
    model,q=author('mirror-rod')
    event=investigate_motion(EventStudy(model.base,(q[1],),EventPolicy(.01,1e-9,1e-10,1e-9)))
    driven=investigate_driven_motion(DrivenStudy(model,q,EventPolicy(.01,1e-9,1e-10,1e-9)))
    import re
    for html,identifier in ((render_event_run(event),'event-data'),(render_driven_run(driven),'driven-data')):
        blob=re.search(r'<script id="'+identifier+r'" type="application/json">(.*?)</script>',html,re.S).group(1)
        data=json.loads(blob)
        assert data['source_geometry']==json.loads(json.dumps(source_geometry(model.spec,model.base.source_annotations)))


def test_stable_geometry_ledger_does_not_relax_field_laws():
    model,_=author('coaxial-rings')
    original=model.as_dict()
    g=source_geometry(model.spec,model.base.source_annotations)
    assert original==model.as_dict()
    assert model.spec.field.sources[0].electric_field_at((.18,0.,0.))
    with pytest.raises(UnsupportedProblem, match='axis'):
        model.spec.field.sources[0].electric_field_at((.18,.01,0.))
    assert g['entries'][0]['status']=='orthographic-projection'
