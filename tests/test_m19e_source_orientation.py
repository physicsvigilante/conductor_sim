"""M19e opt-in source IDs and world-axis orientation; physics never changes."""
from __future__ import annotations

from copy import deepcopy
import json
from math import isclose
from pathlib import Path
import os
from hashlib import sha256
import subprocess
import sys

import pytest

from conductor_sim import (
    PointCharge, PrescribedField, TestChargeMotionSpec, UniformDiskAxisField,
    UniformRingAxisField, UniformElectricField, UniformSphereField,
    integrate_test_charge, sample_test_charge_viewer,
)
from conductor_sim.prescribed_viewer_validation import validate_prescribed_viewer
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.viewer import render_html
from examples.ring_source_orientation import SOURCE_ANNOTATIONS, make_case

ROOT=Path(__file__).resolve().parents[1]


def bundles():
    spec=make_case()
    result=integrate_test_charge(spec)
    plain=sample_test_charge_viewer(spec,result=result,visual_frames=8,profile_samples=101)
    annotated=sample_test_charge_viewer(spec,result=result,visual_frames=8,profile_samples=101,
              axis_orientation='horizontal',source_annotations=SOURCE_ANNOTATIONS)
    return spec,result,plain,annotated


def test_real_physical_axis_is_horizontal_without_any_physics_change():
    spec,result,plain,new=bundles()
    assert plain['frames'][0]['plane']['v']==pytest.approx((1.,0.,0.))
    assert new['frames'][0]['plane']['u']==pytest.approx((1.,0.,0.))
    assert new['frames'][0]['plane']['v']==pytest.approx((0.,1.,0.))
    assert new['prescribed_motion']['axis_display']=='horizontal'
    assert 'axis_display' not in plain['prescribed_motion']
    assert 'source_annotations' not in plain['prescribed_motion']
    physics_keys=('field','samples','events','termination','profile_samples',
                  'max_abs_energy_drift_j','energy_convention','initial_velocity_m_s')
    for key in physics_keys:
        assert new['prescribed_motion'][key]==plain['prescribed_motion'][key]
    assert [f['charges'][0]['position_m'] for f in new['frames']]==[f['charges'][0]['position_m'] for f in plain['frames']]
    assert new['prescribed_motion']['graph_sample_count']==len(result.samples)
    assert json.loads(json.dumps(result.spec.field.as_dict()))==new['prescribed_motion']['field']
    assert spec.field.sources[0].axis==(1.,0.,0.)


def test_source_identity_metadata_is_authored_and_stable_across_hypotheticals():
    spec,result,plain,new=bundles()
    expected=[{'source_index':i,'id':identity,'label':label,
               'kind':type(src).__name__}
              for i,((identity,label),src) in enumerate(zip(SOURCE_ANNOTATIONS,spec.field.sources))]
    assert new['prescribed_motion']['source_annotations']==expected
    axial=make_case('axial-field')
    alt=sample_test_charge_viewer(axial,axis_orientation='horizontal',
                                  source_annotations=SOURCE_ANNOTATIONS,visual_frames=8,profile_samples=101)
    assert alt['prescribed_motion']['source_annotations']==expected
    assert new['prescribed_motion']['field']!=alt['prescribed_motion']['field']
    # The display does not interpret IDs as instructions or create a source registry.
    assert set(new.keys())==set(plain.keys())
    assert new['prescribed_motion']['particle']==plain['prescribed_motion']['particle']
    assert validate_prescribed_viewer(new) is True


def test_axial_profile_and_graph_values_invariant_to_screen_rotation():
    _,_,old,new=bundles()
    for a,b in zip(old['prescribed_motion']['profile_samples'],new['prescribed_motion']['profile_samples']):
        assert a==b
    axis=new['prescribed_motion']['axis']
    assert axis==[1.,0.,0.]
    rows=new['prescribed_motion']['samples']
    assert all(isclose(r['displacement_m'][0],sum(r['displacement_m'][i]*axis[i] for i in range(3)),abs_tol=1e-12)
               for r in rows)
    assert 'prescribed.axis' in render_html(new)
    assert 'no off-axis' in new['prescribed_motion']['display_note'] or 'not a sampled field map' in new['prescribed_motion']['display_note']


def test_ring_and_disc_axis_projection_rotated_but_discontinuity_preserved():
    from math import sqrt
    tilted=UniformRingAxisField(1e-9,.1,center_m=(.3,.4,.5),axis=(1.,1.,0.))
    a=1/sqrt(2)
    spec=TestChargeMotionSpec(PrescribedField((tilted,)),PointCharge(1e-9,(.3+.2*a,.4+.2*a,.5)),
         1e-6,(-.2*a,-.2*a,0.),.1,.01,plane_normal=(0.,0.,1.),length_scale_m=.1)
    result=integrate_test_charge(spec)
    view=sample_test_charge_viewer(spec,result=result,axis_orientation='horizontal',profile_samples=101)
    plane=view['frames'][0]['plane']
    assert plane['u']==pytest.approx((a,a,0.))
    assert plane['v']==pytest.approx((-a,a,0.))
    assert view['prescribed_motion']['profile_samples'][50]['potential_v']==pytest.approx(tilted.potential_at(tilted.center_m))
    assert validate_prescribed_viewer(view)

    disc=UniformDiskAxisField(1e-11,.1,axis=(1.,0.,0.))
    spec=TestChargeMotionSpec(PrescribedField((disc,)),PointCharge(-1e-9,(.2,0.,0.)),
         1e-6,(-.01,0.,0.),.01,.005,length_scale_m=.1,source_guard_m=1e-5)
    b=sample_test_charge_viewer(spec,axis_orientation='horizontal',profile_samples=101)
    assert b['frames'][0]['plane']['u']==pytest.approx((1.,0.,0.))
    assert b['prescribed_motion']['profile_samples'][50]['coordinate_m']==pytest.approx(0.,abs=1e-15)
    assert b['prescribed_motion']['profile_samples'][50]['field_parallel_v_m'] is None
    assert b['prescribed_motion']['profile_samples'][50]['potential_v'] is not None
    assert validate_prescribed_viewer(b)


def test_nonaxial_field_cannot_claim_ring_axis_oriented_display():
    spec=TestChargeMotionSpec(PrescribedField((UniformElectricField((0.,0.,-1.)),)),
        PointCharge(1e-9,(0.,0.,0.)),1e-6,(.2,0.,0.),.02,.01)
    with pytest.raises(UnsupportedProblem,match='ring/disc'):
        sample_test_charge_viewer(spec,axis_orientation='horizontal')
    assert 'axis_display' not in sample_test_charge_viewer(spec)['prescribed_motion']
    with pytest.raises(ValueError,match='vertical or horizontal'):
        sample_test_charge_viewer(spec,axis_orientation='upside-down')


@pytest.mark.parametrize('annotations,match',[
    ([('ring:A','Ring A')], 'one ID/label'),
    ([('ring:A','Ring A'),('ring:A','Ring B'),('field:external','External field')], 'unique'),
    ([('ring:A','Ring A'),('ring:B','Ring A'),('field:external','External field')], 'unique'),
    ([('ring A','Ring A'),('ring:B','Ring B'),('field:external','External field')], 'ASCII'),
    ([('ring:A','Ring A'),('ring:B','Ring B'),('field:external','')], 'display label'),
    ([('ring:A','Ring A'),('ring:B','Ring B\n'),('field:external','External field')], 'display label'),
    ([('ring:A','Ring A'),('ring:B',' Ring B'),('field:external','External field')], 'display label'),
    ([('ring:A','Ring A'),('ring:B','Ring B'),('field:external','x'*49)], 'display label'),
    ([('ring:A','Ring A'),('ring:B','Ring B'),('field:external',2)], 'display label'),
    ([('ring:A','Ring A'),('ring:B','Ring B'),('field:external',)], 'entries'),
])
def test_invalid_identity_annotations_refused_before_any_html(annotations,match):
    spec=make_case()
    with pytest.raises(ValueError,match=match):
        sample_test_charge_viewer(spec,source_annotations=annotations)


@pytest.mark.parametrize('mutation,match',[
    (lambda b: b['prescribed_motion'].update(axis_display='diagonal'), 'axis orientation'),
    (lambda b: b['prescribed_motion'].update(axis_display='vertical'), 'display axis orientation'),
    (lambda b: b['prescribed_motion'].pop('axis_display'), 'display axis orientation'),
    (lambda b: b['prescribed_motion']['source_annotations'][0].update(source_index=1), 'source annotation'),
    (lambda b: b['prescribed_motion']['source_annotations'][1].update(kind='FixedPointField'), 'source annotation'),
    (lambda b: b['prescribed_motion']['source_annotations'][1].update(id='ring:A'), 'duplicate'),
    (lambda b: b['prescribed_motion']['source_annotations'].pop(), 'source annotations'),
    (lambda b: b['prescribed_motion']['source_annotations'][0].update(label='Ring\nA'), 'source annotation'),
    (lambda b: b['prescribed_motion']['field']['sources'][0]['axis'].__setitem__(0,.9999999), 'axis|inventory'),
    (lambda b: b['frames'][0]['plane'].update(u=[0.,0.,1.]), 'display|invalid|axes'),
])
def test_tampered_axis_and_source_metadata_fail_closed(mutation,match):
    _,_,_,b=bundles();b=deepcopy(b)
    mutation(b)
    with pytest.raises((ValueError,TypeError),match=match):
        render_html(b)


@pytest.mark.parametrize('case,expected_sha256',[
    ('two-rings', '078472f7069e9194efd8b97e46e9d43d302485c756d69457442c86d5ac091877'),
    ('uniform-deflection', '4c451cc8b43aa89597a3ac0c6ae7f85db5aca604e8f0e29189347dbede65b200'),
    ('disc-approach', '729980813c9a6e7e638259f5d1c3ba6b6ecfdc5af6868037930859c9fed4cb49'),
    ('ring-turning', 'eb0804e60110620cb72387dc8dc8b5c78318ceb194e63808366d441bfb7a1a2f'),
])
def test_old_default_bundle_matches_accepted_m19b_frozen_sha256(case,expected_sha256):
    """The SHA-256 constants were generated from the accepted M19b source tree
    751fb595f598bf134d86bdbc1f3c8dfb0b529626 BEFORE M19e changes.
    Checking them here requires no historical checkout on GitHub CI.
    """
    from examples.test_charge_motion import make_case
    bundle=sample_test_charge_viewer(make_case(case),visual_frames=7,
                                      profile_samples=101,resolution=13)
    text=json.dumps(bundle,sort_keys=True,separators=(',',':'),allow_nan=False)
    assert sha256(text.encode('utf-8')).hexdigest()==expected_sha256


def test_source_label_html_is_text_not_executable_markup():
    spec=make_case()
    b=sample_test_charge_viewer(spec,source_annotations=(
        ('ring:A','Ring <A>'),('ring:B','Ring B & C'),('field:external','External field')))
    html=render_html(b)
    assert '<script>alert' not in html
    assert '\\u003c' in html
    assert validate_prescribed_viewer(b)


def test_readable_teacher_example_two_cases_export_html_and_json(tmp_path):
    for case in ('field-off','axial-field'):
        output=tmp_path/(case+'.html')
        proc=subprocess.run([sys.executable,'examples/ring_source_orientation.py',
              '--case',case,'--output',str(output)],cwd=ROOT,
              env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT)),
              text=True,capture_output=True,timeout=45)
        assert proc.returncode==0,proc.stderr
        sidecar=json.loads(output.with_suffix('.json').read_text())
        html=output.read_text()
        embedded=html.split('<script id="simulation-data" type="application/json">',1)[1].split('</script>',1)[0]
        assert json.loads(embedded)==sidecar
        assert sidecar['prescribed_motion']['axis_display']=='horizontal'
        assert 'teacher hypothetical' in ' '.join(sidecar['frames'][0]['notes']).lower()
        assert sidecar['prescribed_motion']['source_annotations'][0]['id']=='ring:A'
        assert sidecar['prescribed_motion']['source_annotations'][1]['id']=='ring:B'
        assert sidecar['prescribed_motion']['termination']=='duration'
        assert validate_prescribed_viewer(sidecar)
