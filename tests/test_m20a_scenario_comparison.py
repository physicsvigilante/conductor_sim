"""M20a: two checked independent scenarios, one physical clock, no browser ODE."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import json
import subprocess
import sys

import pytest

from conductor_sim import (PrescribedField, TestChargeMotionSpec as MotionSpec, sample_test_charge_viewer,
    prepare_scenario_comparison, render_scenario_comparison,
    write_scenario_comparison)
from conductor_sim.viewer import render_html
from conductor_sim.prescribed_viewer_validation import validate_prescribed_viewer
from examples.compare_prescribed_scenarios import make_pair, make_bundles
from examples.ring_source_orientation import SOURCE_ANNOTATIONS


@lru_cache(maxsize=8)
def pair(case):
    return make_bundles(case,frames=12,profile_samples=101,resolution=13)


@pytest.mark.parametrize('case,property_name,identity',[
    ('axial-field','field_v_m','field:external'),
    ('double-mass','mass_kg','test-particle'),
    ('ring-b-shift','center_m','ring:B'),
    ('gauge-shift','reference_potential_v','field:external'),
])
def test_four_real_teacher_revisions_retain_source_identites_and_full_samples(case,property_name,identity):
    before,after,_=pair(case)
    m=prepare_scenario_comparison(before,after,
        baseline_id='ring-base',variation_id='ring-'+case,
        baseline_provenance='Teacher hypothetical: field removed, mass stipulated.',
        variation_provenance='Teacher hypothetical: one physical change.')
    assert m['schema']=='conductor_sim.scenario_comparison.v1'
    assert m['scenario_ids']==['ring-base','ring-'+case]
    assert m['time_interpolation'] is False
    assert m['physical_sample_count']==126
    assert len(m['frame_times_s'])==12
    assert m['frame_times_s'][0]==0. and m['frame_times_s'][-1]==.25
    assert [x['component_id'] for x in m['revisions']]==[identity]
    assert [x['property'] for x in m['revisions']]==[property_name]
    assert len(m['source_ids'])==3
    assert set(m['source_ids'])=={'ring:A','ring:B','field:external'}
    assert before['prescribed_motion']['field']['sources'][0]==after['prescribed_motion']['field']['sources'][0]
    assert all(b['time_s']==v['time_s'] for b,v in zip(
        before['prescribed_motion']['samples'],after['prescribed_motion']['samples']))
    assert before['frames'][0]['plane']==after['frames'][0]['plane']
    assert validate_prescribed_viewer(before) and validate_prescribed_viewer(after)


def test_gauge_shift_is_strictly_displayed_as_reference_change_not_motion():
    a,b,_=pair('gauge-shift')
    ra,rb=a['prescribed_motion']['samples'],b['prescribed_motion']['samples']
    assert len(ra)==len(rb)
    q=a['prescribed_motion']['particle']['charge_c']
    for x,y in zip(ra,rb):
        assert x['position_m']==y['position_m']
        assert x['velocity_m_s']==y['velocity_m_s']
        assert x['force_n']==y['force_n']
        assert y['potential_v']-x['potential_v']==pytest.approx(17.,abs=1e-9)
        assert y['potential_energy_j']-x['potential_energy_j']==pytest.approx(q*17.,abs=1e-12)
    m=prepare_scenario_comparison(a,b)
    for x,y in m['position_m']:
        assert x==y


def test_mass_and_source_geometry_changes_create_genuine_different_paths():
    for case in ('double-mass','ring-b-shift','axial-field'):
        a,b,_=pair(case)
        p=a['prescribed_motion']['samples'][-1]['position_m']
        q=b['prescribed_motion']['samples'][-1]['position_m']
        assert abs(p[0]-q[0])>1e-4
        assert a['prescribed_motion']['energy_convention']==b['prescribed_motion']['energy_convention']


def test_comparison_html_preserves_both_independent_renderers_and_source_arrays():
    a,b,_=pair('axial-field')
    aa,bb=deepcopy(a),deepcopy(b)
    page=render_scenario_comparison(a,b,title='Two separately simulated rings')
    assert page.startswith('<!doctype html>')
    assert page.count('<iframe')==2
    assert 'no interpolated physics or browser integration' in page
    assert 'ring:A' in page and 'ring:B' in page
    assert 'field_v_m' in page
    assert 'conductor_sim.scenario_comparison.v1' in page
    assert 'conductor_sim.test_charge_viewer.v1' in page
    assert 'compare-time' in page and 'compare-delta' in page
    assert a==aa and b==bb
    assert render_html(a) and render_html(b)
    assert all(x['field']['sources']==y['field']['sources'] for x,y in [
        (a['prescribed_motion'],a['prescribed_motion']),
        (b['prescribed_motion'],b['prescribed_motion'])])


def test_write_html_is_reproducible_and_directory_created(tmp_path):
    a,b,_=pair('double-mass')
    target=tmp_path/'deep'/'teaching.html'
    written=write_scenario_comparison(a,b,target,variation_provenance='Teacher example: double mass')
    assert written==target and target.is_file()
    first=target.read_bytes()
    write_scenario_comparison(a,b,target,variation_provenance='Teacher example: double mass')
    assert target.read_bytes()==first
    assert b'play' in first


def test_no_annotations_refuses_unproven_source_identity():
    spec=make_pair('axial-field')[0]
    no_id=sample_test_charge_viewer(spec,visual_frames=12,resolution=13,profile_samples=101,
                                    extent_m=2.25,axis_orientation='horizontal')
    with pytest.raises(ValueError,match='source IDs'):
        prepare_scenario_comparison(no_id,pair('axial-field')[1])


def test_distinct_input_object_required():
    a,_,_=pair('axial-field')
    with pytest.raises(ValueError,match='distinct'):
        prepare_scenario_comparison(a,a)


def test_require_same_certified_world_axes_and_common_scale():
    a,b,_=pair('axial-field')
    original=make_pair('axial-field')[1]
    # Legitimate separately verified output, but automatic scale is different.
    changed_scale=sample_test_charge_viewer(original,visual_frames=12,resolution=13,
        profile_samples=101,source_annotations=SOURCE_ANNOTATIONS,axis_orientation='horizontal')
    with pytest.raises(ValueError,match='identical physical slice'):
        prepare_scenario_comparison(a,changed_scale)
    changed_orientation=sample_test_charge_viewer(original,visual_frames=12,resolution=13,
        profile_samples=101,extent_m=2.25,source_annotations=SOURCE_ANNOTATIONS,
        axis_orientation='vertical')
    with pytest.raises(ValueError,match='orientations'):
        prepare_scenario_comparison(a,changed_orientation)


def test_differing_time_grids_refuse_instead_of_interpolating():
    a,_,_=pair('axial-field')
    spec=replace(make_pair('axial-field')[1],sample_interval_s=.005)
    b=sample_test_charge_viewer(spec,visual_frames=12,resolution=13,profile_samples=101,
        extent_m=2.25,axis_orientation='horizontal',source_annotations=SOURCE_ANNOTATIONS)
    with pytest.raises(ValueError,match='time grids'):
        prepare_scenario_comparison(a,b)
    spec=make_pair('axial-field')[1]
    b=sample_test_charge_viewer(spec,visual_frames=9,resolution=13,profile_samples=101,
        extent_m=2.25,axis_orientation='horizontal',source_annotations=SOURCE_ANNOTATIONS)
    with pytest.raises(ValueError,match='field frame times'):
        prepare_scenario_comparison(a,b)


def test_rejects_invalid_source_id_and_changed_authored_label():
    a,_,_=pair('axial-field')
    spec=make_pair('axial-field')[1]
    changed_ids=(('ring:other','Ring A'),*SOURCE_ANNOTATIONS[1:])
    b=sample_test_charge_viewer(spec,visual_frames=12,resolution=13,profile_samples=101,
        extent_m=2.25,axis_orientation='horizontal',source_annotations=changed_ids)
    with pytest.raises(ValueError,match='identities differ'):
        prepare_scenario_comparison(a,b)
    labels=(('ring:A','Ring A different'),*SOURCE_ANNOTATIONS[1:])
    b=sample_test_charge_viewer(spec,visual_frames=12,resolution=13,profile_samples=101,
        extent_m=2.25,axis_orientation='horizontal',source_annotations=labels)
    with pytest.raises(ValueError,match='label changed'):
        prepare_scenario_comparison(a,b)


def test_source_inventory_cannot_be_disguised_by_metadata_mutation():
    a,b,_=pair('axial-field')
    corrupt=deepcopy(b)
    corrupt['prescribed_motion']['samples'][5]['electric_field_v_m'][0]+=50000
    with pytest.raises(ValueError,match='saved field/potential'):
        prepare_scenario_comparison(a,corrupt)
    corrupt=deepcopy(b)
    corrupt['prescribed_motion']['field']['sources'][0]['charge_c']*=1.1
    with pytest.raises(ValueError,match='saved field/potential'):
        prepare_scenario_comparison(a,corrupt)


def test_provenance_ids_and_html_are_safely_escaped():
    a,b,_=pair('axial-field')
    title='Teacher <script>alert(1)</script> & exploration'
    provenance='Teacher hypothetical: <img src=x onerror=alert(2)> & reference'
    html=render_scenario_comparison(a,b,title=title,
        variation_provenance=provenance,baseline_provenance=provenance)
    assert '<script>alert(1)</script>' not in html
    assert '<img src=x onerror=alert(2)>' not in html
    assert '&lt;script&gt;' in html
    assert '&lt;img src=x' in html
    for kwargs,expected in [({'baseline_id':'bad id'},'identifier'),
                             ({'variation_title':'\n'},'printable'),
                             ({'variation_provenance':''},'nonempty'),
                             ({'variation_id':'baseline'},'distinct')]:
        with pytest.raises(ValueError,match=expected):
            prepare_scenario_comparison(a,b,**kwargs)


def test_original_v1_viewer_unchanged_by_this_new_composition():
    a,b,_=pair('gauge-shift')
    html0=render_html(a)
    prepare_scenario_comparison(a,b)
    assert html0==render_html(a)
    assert 'window.conductorViewer' in html0
    assert 'conductorComparison' not in html0


@pytest.mark.parametrize('case',[ 'axial-field','double-mass','ring-b-shift','gauge-shift'])
def test_cli_exports_openable_offline_comparison(case,tmp_path):
    dest=tmp_path/(case+'.html')
    result=subprocess.run([sys.executable, 'examples/compare_prescribed_scenarios.py',
        '--case',case,'--output',str(dest)],cwd=Path(__file__).resolve().parents[1],
        capture_output=True,text=True,check=True)
    assert dest.exists() and dest.stat().st_size>100000
    assert 'No browser physics' in result.stdout


def test_reusable_comparison_also_supports_nonaxial_uniform_field_motion():
    # This is an independent source family, not an M19e axial-ring hardcode.
    from examples.test_charge_motion import make_case
    spec=make_case('uniform-deflection')
    modified=replace(spec,mass_kg=2*spec.mass_kg)
    def view(s):
        return sample_test_charge_viewer(s,visual_frames=9,resolution=13,
            extent_m=.5,source_annotations=(('field:U','Applied uniform field'),))
    left,right=view(spec),view(modified)
    m=prepare_scenario_comparison(left,right)
    assert m['source_ids']==['field:U']
    assert m['revisions'][0]['property']=='mass_kg'
    assert m['revisions'][0]['component_id']=='test-particle'
    assert m['frame_times_s'][-1]==.4
    assert m['plane']==left['frames'][0]['plane']==right['frames'][0]['plane']
    assert left['prescribed_motion']['profile_mode']=='elapsed-time'
    assert right['prescribed_motion']['profile_mode']=='elapsed-time'
    assert m['position_m'][-1][0]!=m['position_m'][-1][1]


def test_real_source_guard_trajectories_are_not_misrepresented_as_complete():
    from conductor_sim import PointCharge, UniformShellField
    field=PrescribedField((UniformShellField(1e-9,.1),))
    spec=MotionSpec(field,PointCharge(1e-9,(0.,0.,0.)),1e-6,
        (.4,0.,0.),.4,.01,length_scale_m=.1,source_guard_m=1e-4)
    def view(s):
        return sample_test_charge_viewer(s,visual_frames=12,resolution=13,
            extent_m=.3,source_annotations=(('shell:X','Thin shell X'),))
    left,right=view(spec),view(replace(spec,mass_kg=2e-6))
    assert left['prescribed_motion']['termination']=='source-guard'
    assert right['prescribed_motion']['termination']=='source-guard'
    assert left['prescribed_motion']['samples'][-1]['time_s']<.4
    assert validate_prescribed_viewer(left) and validate_prescribed_viewer(right)
    with pytest.raises(ValueError,match='complete duration histories'):
        prepare_scenario_comparison(left,right)
