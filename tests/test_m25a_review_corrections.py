"""Post-Chat-B M25a review: independently motivated F-01/F-02/F-03 regressions.

Preserves reviewed head 31ee492b... and untouched reviewer findings. Tests
qualified actual models rather than altering unqualified exported HTML.
"""
from __future__ import annotations

from dataclasses import replace
from math import cos, hypot, pi, sin, sqrt

import pytest

from conductor_sim import (
    PointCharge, PrescribedField, TestChargeMotionSpec,
    UniformRingAxisField, UniformSphereField, UniformShellField,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.driven_motion import (
    DrivenModel, DrivenStudy, investigate_driven_motion,
)
from conductor_sim.driven_viewer import render_driven_run
from conductor_sim.motion_events import EventPolicy, EventStudy, investigate_motion
from conductor_sim.motion_event_viewer import render_event_run
from conductor_sim.source_geometry import source_geometry, _projected_major_axis
from examples.public_api_event_trial import author


def _qualified_ring_view(angle):
    model, questions = author('coaxial-rings')
    normal = (0., cos(angle), sin(angle))
    rotated = replace(model.base.spec, plane_normal=normal)
    base = PrescribedScenario('review:rotated-ring-view', rotated, model.base.source_annotations,
        'Teacher hypothetical: actual axis-contained plane viewed after 3-D world rotation')
    driven = DrivenModel(base, model.drag, model.applied,
        'review:rotated-ring-driven', 'Retained source laws and true physical plane')
    return base, driven, questions


@pytest.mark.parametrize('angle', [0., pi/4, pi/2, 3*pi/4, pi])
def test_f01_true_major_axis_on_qualified_rotated_ring_plane(angle):
    base, driven, questions = _qualified_ring_view(angle)
    original = driven.as_dict()
    g = source_geometry(base.spec, base.source_annotations, trusted_scenario=base)
    for e, src in zip(g['entries'], base.spec.field.sources):
        assert e['shape'] == 'ring-projection' and e['edge_on']
        assert hypot(*e['projected_major_axis_uv']) == pytest.approx(src.radius_m, abs=2e-14)
        # Independent projection from the actual ring in world yz coordinates.
        expected = [tuple(sum((src.center_m[j] - g['origin_m'][j] +
            (0., src.radius_m*cos(t), src.radius_m*sin(t))[j])*basis[j] for j in range(3))
            for basis in (g['u'],g['v'])) for t in (2*pi*k/720 for k in range(721))]
        center = e['center_uv']
        assert max(hypot(p[0]-center[0],p[1]-center[1]) for p in expected) == pytest.approx(src.radius_m,abs=2e-14)
        # The edge-on line endpoints agree with actual sampled rim extrema.
        z=e['projected_major_axis_uv']; zlen=hypot(*z)
        assert max(abs((p[0]-center[0])*z[0]/zlen+(p[1]-center[1])*z[1]/zlen)
                   for p in expected)==pytest.approx(zlen,abs=2e-14)
    assert driven.as_dict() == original


def test_f01_45_degree_geometry_is_nontrivial_and_would_fail_old_marker_rule():
    base,_,_= _qualified_ring_view(pi/4)
    e=source_geometry(base.spec,base.source_annotations)['entries'][0]
    # Both old basis columns are only R/sqrt(2), but the true major semi-axis is R.
    assert max(hypot(*a) for a in e['rim_basis_uv'])==pytest.approx(.3/sqrt(2),rel=1e-14)
    assert hypot(*e['projected_major_axis_uv'])==pytest.approx(.3,rel=1e-14)
    assert hypot(*e['projected_major_axis_uv'])-max(hypot(*a) for a in e['rim_basis_uv']) > .08


@pytest.mark.parametrize('a,b,r',[
    ((.13,0.),(0.,.13),.13),  # face-on pure geometry; no axial motion claim
    ((.12,0.),(0.,.08),.12),  # elliptical pure geometry
    ((.2/sqrt(2),0.),(.2/sqrt(2),0.),.2),  # edge-on 45deg parameter basis
    ((.0,.25/sqrt(2)),(.0,-.25/sqrt(2)),.25),
    ((0.,0.),(0.,.1),.1),
])
def test_f01_projector_major_axis_numeric_face_on_ellipse_edge_on(a,b,r):
    z=_projected_major_axis(a,b,r)
    # Independent numerical maximization of |a cos(t)+b sin(t)|.
    samples=[hypot(a[0]*cos(2*pi*k/4096)+b[0]*sin(2*pi*k/4096),
                   a[1]*cos(2*pi*k/4096)+b[1]*sin(2*pi*k/4096)) for k in range(4096)]
    assert hypot(*z)==pytest.approx(max(samples),rel=2e-6)


@pytest.mark.parametrize('kind', [UniformSphereField, UniformShellField])
@pytest.mark.parametrize('h,expected',[(4e-12,'slice-intersection'),(5e-12,'slice-tangent'),(8e-12,'omitted-out-of-plane')])
def test_f02_actual_slice_classification_even_inside_solver_plane_tolerance(kind,h,expected):
    radius=5e-12
    source=kind(1e-10,radius,(.2,h,0.))
    spec=TestChargeMotionSpec(PrescribedField((source,)),PointCharge(-1e-9,(0.,0.,0.)),
        2e-9,(.01,0.,0.),duration_s=.15,sample_interval_s=.05,
        plane_normal=(0.,1.,0.),length_scale_m=1.,source_guard_m=.001)
    e=source_geometry(spec,(('source:radial','Very small physical source'),))['entries'][0]
    assert e['status']==expected
    if h>radius:
        assert e['shape']=='omitted' and e['bounds_uv']==[]
        assert 'No physical intersection' in e['notes']
    elif h==radius:
        assert e['shape']=='slice-tangent' and e['slice_radius_m']==0.
        assert e['bounds_uv']==[e['center_uv']]
        assert 'single-point tangent' in e['notes']
    else:
        assert e['shape']==('sphere-slice' if kind is UniformSphereField else 'shell-slice')
        assert e['slice_radius_m']==pytest.approx(3e-12,rel=1e-14)
        assert e['slice_radius_m']>0


def test_f03_author_supplied_ids_not_independently_authenticated_but_integrated_binding_is():
    model,_=author('coaxial-rings')
    saved=model.as_dict()
    base=model.base
    swapped=tuple(reversed(base.source_annotations))
    # Raw helper is explicitly an authored assertion, not authenticated provenance.
    raw=source_geometry(base.spec,swapped)
    assert [e['source_id'] for e in raw['entries']]==['ring:B','ring:A']
    with pytest.raises(ValueError,match='retained scenario binding'):
        source_geometry(base.spec,swapped,trusted_scenario=base)
    with pytest.raises(ValueError,match='retained scenario binding'):
        source_geometry(replace(base.spec,sample_interval_s=.06),base.source_annotations,trusted_scenario=base)
    bound=source_geometry(base.spec,base.source_annotations,trusted_scenario=base)
    assert [e['source_id'] for e in bound['entries']]==['ring:A','ring:B']
    assert model.as_dict()==saved


def test_f03_both_public_viewers_preserve_same_bound_ids_without_changing_physics():
    base,driven,qs=_qualified_ring_view(pi/4)
    policy=EventPolicy(.01,1e-9,1e-10,1e-9)
    event_run=investigate_motion(EventStudy(base,(qs[1],),policy))
    driven_run=investigate_driven_motion(DrivenStudy(driven,qs,policy))
    before_event=event_run.as_dict();before_driven=driven_run.as_dict()
    a=render_event_run(event_run);b=render_driven_run(driven_run)
    assert 'projected_major_axis_uv' in a and 'projected_major_axis_uv' in b
    assert "function drawSourceGeometry(" in a and "function drawSourceGeometry(" in b
    assert event_run.as_dict()==before_event and driven_run.as_dict()==before_driven
