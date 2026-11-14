"""M19f: compare independently verified saved M19e sources, no new solver."""
from copy import deepcopy
from dataclasses import replace
import json
from math import hypot, isclose

import pytest

from conductor_sim import (
    PointCharge, PrescribedField, TestChargeMotionSpec as MotionSpec, UniformElectricField,
    compare_prescribed_scenarios, sample_prescribed_scenario_comparison,
    sample_test_charge_viewer, validate_comparison,
    render_scenario_comparison_html, write_scenario_comparison,
    integrate_test_charge,
)
from examples.ring_source_orientation import make_case, SOURCE_ANNOTATIONS


@pytest.fixture(scope='module')
def comparison():
    return sample_prescribed_scenario_comparison(
        make_case('field-off'), make_case('axial-field'),
        source_annotations=SOURCE_ANNOTATIONS,
        labels=('Field off','Axial field on'), visual_frames=12,
        profile_samples=101, source_context='Two-ring teacher hypothetical')


def test_physical_source_and_model_changes_are_derived_not_invented(comparison):
    assert comparison['schema']=='conductor_sim.prescribed_comparison.v1'
    assert comparison['source_ids']==['ring:A','ring:B','field:external']
    assert comparison['unchanged_source_values']==['ring:A','ring:B']
    assert comparison['unchanged_particle_properties']==['charge_c','position_m']
    assert comparison['changes']==[dict(owner='field:external',property='field_v_m',
                                       before=[0.,0.,0.],after=[100000.,0.,0.])]
    assert len(comparison['samples'])==12
    assert len(comparison['scenario_bundles'])==2
    assert all(row['time_s']==t for row,t in zip(comparison['samples'],comparison['time_s']))
    assert comparison['time_s'][0]==0
    assert comparison['time_s'][-1]==pytest.approx(.25)
    assert comparison['samples'][-1]['delta_position_norm_m']>0
    assert comparison['samples'][0]['delta_position_norm_m']==0
    assert comparison['scenario_bundles'][0]['frames'][0]['plane']==comparison['scenario_bundles'][1]['frames'][0]['plane']
    assert validate_comparison(comparison)==comparison


def test_two_real_saved_trajectories_have_no_new_solution_or_browser_time_field(comparison):
    left,right=(b['prescribed_motion'] for b in comparison['scenario_bundles'])
    assert len(left['samples'])==len(right['samples'])==126
    assert left['field']['sources'][0]==right['field']['sources'][0]
    assert left['field']['sources'][1]==right['field']['sources'][1]
    assert left['field']['sources'][2]['field_v_m']==[0,0,0]
    assert right['field']['sources'][2]['field_v_m']==[100000,0,0]
    for ia,ib,comp in zip(left['frame_sample_indices'],right['frame_sample_indices'],comparison['samples']):
        a,b=left['samples'][ia],right['samples'][ib]
        assert comp['left_sample_index']==ia and comp['right_sample_index']==ib
        assert a['time_s']==b['time_s']==comp['time_s']
        d=[y-x for x,y in zip(a['position_m'],b['position_m'])]
        assert comp['delta_position_m']==d
        assert comp['delta_position_norm_m']==pytest.approx(hypot(*d))
        assert comp['delta_kinetic_energy_j']==b['kinetic_energy_j']-a['kinetic_energy_j']
        assert comp['delta_total_energy_j']==b['total_energy_j']-a['total_energy_j']
        for row in (a,b):
            assert row['potential_energy_j']==row['interaction_energy_j']==pytest.approx(
                row['particle_potential_energy_j'] if 'particle_potential_energy_j' in row else row['potential_energy_j'])
            assert row['total_energy_j']==pytest.approx(row['kinetic_energy_j']+row['potential_energy_j'])


def test_supplied_trajectories_are_never_reintegrated(monkeypatch):
    import conductor_sim.scenario_comparison as sc
    specs=[make_case('field-off'),make_case('axial-field')]
    results=[integrate_test_charge(s) for s in specs]
    def forbidden(*_,**__):
        raise AssertionError('second integration occurred')
    monkeypatch.setattr('conductor_sim.prescribed_motion.integrate_test_charge',forbidden)
    monkeypatch.setattr('conductor_sim.prescribed_viewer.integrate_test_charge',forbidden)
    result=sc.sample_prescribed_scenario_comparison(*specs,
        result_left=results[0],result_right=results[1],
        source_annotations=SOURCE_ANNOTATIONS)
    assert len(result['samples'])==12


def test_identical_scenario_produces_no_false_changes():
    spec=make_case('field-off')
    result=integrate_test_charge(spec)
    both=sample_prescribed_scenario_comparison(spec,spec,
        result_left=result,result_right=result,source_annotations=SOURCE_ANNOTATIONS)
    assert both['changes']==[]
    assert both['unchanged_source_values']==['ring:A','ring:B','field:external']
    assert all(row['delta_position_norm_m']==0 for row in both['samples'])


def test_mass_variation_uses_scenario_physics_not_new_force_law():
    a=make_case('field-off')
    b=replace(a,mass_kg=.2)
    comparison=sample_prescribed_scenario_comparison(a,b,source_annotations=SOURCE_ANNOTATIONS,
                                                      labels=('Base','Mass 2x'))
    assert comparison['changes']==[dict(owner='motion',property='mass_kg',before=.1,after=.2)]
    assert len(comparison['unchanged_source_values'])==3
    assert comparison['samples'][-1]['delta_position_norm_m']>0


def test_voltage_reference_changes_energy_not_motion():
    a=make_case('field-off')
    f=list(a.field.sources)
    f[-1]=UniformElectricField((0.,0.,0.),reference_potential_v=17.)
    b=replace(a,field=PrescribedField(tuple(f)))
    c=sample_prescribed_scenario_comparison(a,b,source_annotations=SOURCE_ANNOTATIONS)
    assert c['changes']==[dict(owner='field:external',property='reference_potential_v',before=0.,after=17.)]
    assert all(x['delta_position_norm_m']==0 for x in c['samples'])
    assert all(x['delta_speed_m_s']==0 for x in c['samples'])
    assert all(x['delta_potential_energy_j']==pytest.approx(a.particle.charge_c*17) for x in c['samples'])
    assert 'gauge' in c['limitations']


@pytest.mark.parametrize('wrong',[
    'not-a-dict','empty','no-schema','missing-bundle','corrupt-force',
    'corrupt-energy','change-times','change-source-id','change-source-kind',
    'drop-ann','change-source-order','change-axis','change-extent',
    'change-frame-index','alter-metric','alter-changes','alter-source-list',
    'wrong-schema','nan-in-source',
])
def test_corrupt_or_incompatible_comparisons_are_explicitly_rejected(comparison,wrong):
    c=deepcopy(comparison)
    a,b=c['scenario_bundles']
    m=b['prescribed_motion']
    if wrong=='not-a-dict': c=[]
    elif wrong=='empty': c={}
    elif wrong=='no-schema': c.pop('schema')
    elif wrong=='missing-bundle': c['scenario_bundles'].pop()
    elif wrong=='corrupt-force': m['samples'][1]['force_n'][0]+=1
    elif wrong=='corrupt-energy': m['samples'][10]['potential_energy_j']+=10
    elif wrong=='change-times': m['samples'][m['frame_sample_indices'][4]]['time_s']+=.002
    elif wrong=='change-source-id': m['source_annotations'][0]['id']='ring:other'
    elif wrong=='change-source-kind': m['source_annotations'][0]['kind']='FixedPointField'
    elif wrong=='drop-ann': m.pop('source_annotations')
    elif wrong=='change-source-order': m['source_annotations'][0],m['source_annotations'][1]=m['source_annotations'][1],m['source_annotations'][0]
    elif wrong=='change-axis': b['frames'][0]['plane']['u']=[0,0,1]
    elif wrong=='change-extent': b['frames'][0]['plane']['x_range_m'][1]+=1
    elif wrong=='change-frame-index': m['frame_sample_indices'][1]+=1
    elif wrong=='alter-metric': c['samples'][1]['delta_position_norm_m']+=1
    elif wrong=='alter-changes': c['changes']=[]
    elif wrong=='alter-source-list': c['source_ids']=[]
    elif wrong=='wrong-schema': c['schema']='invalid'
    elif wrong=='nan-in-source': m['field']['sources'][0]['charge_c']=float('nan')
    with pytest.raises((ValueError,TypeError)):
        validate_comparison(c)


@pytest.mark.parametrize('alter',[
    'time-disagreement','source-id-disagreement','missing-identity','different-extent',
    'different-slice','different-profiles',
])
def test_independent_valid_source_bundles_must_be_comparable(comparison,alter):
    a,b=deepcopy(comparison['scenario_bundles'])
    if alter=='time-disagreement':
        # Valid independent trajectory with a different output interval.
        spec=replace(make_case('axial-field'),sample_interval_s=.005)
        b=sample_test_charge_viewer(spec,source_annotations=SOURCE_ANNOTATIONS,
                                    axis_orientation='horizontal',visual_frames=12,
                                    profile_samples=101,extent_m=a['frames'][0]['plane']['x_range_m'][1])
    elif alter=='source-id-disagreement': b['prescribed_motion']['source_annotations'][1]['id']='ring:C'
    elif alter=='missing-identity': b['prescribed_motion'].pop('source_annotations')
    elif alter=='different-extent':
        spec=make_case('axial-field')
        b=sample_test_charge_viewer(spec,source_annotations=SOURCE_ANNOTATIONS,
            axis_orientation='horizontal',visual_frames=12,
            profile_samples=101,extent_m=3.0)
    elif alter=='different-slice':
        spec=make_case('axial-field')
        b=sample_test_charge_viewer(spec,source_annotations=SOURCE_ANNOTATIONS,
            axis_orientation='vertical',visual_frames=12,profile_samples=101)
    elif alter=='different-profiles':
        # A legitimate *non-axial* uniform-field run is an elapsed-time profile.
        spec=MotionSpec(PrescribedField((UniformElectricField((0.,0.,100.)),)),
            PointCharge(1e-9,(0.,0.,0.)),.1,(.1,0.,0.),.25,.002,length_scale_m=.3)
        b=sample_test_charge_viewer(spec,source_annotations=(('field:external','External field'),),
                                    visual_frames=12,profile_samples=101)
    with pytest.raises(ValueError):
        compare_prescribed_scenarios(a,b)


@pytest.mark.parametrize('labels,context',[
    (('Same','Same'),'Normal'),
    (('A\nB','C'),'Normal'),
    (('A','B'),''),
    (('A',),'Normal'),
    (('A'*101,'B'),'Normal'),
])
def test_reject_invalid_presentation_labels(comparison,labels,context):
    with pytest.raises(ValueError):
        compare_prescribed_scenarios(*comparison['scenario_bundles'],
                                     labels=labels,source_context=context)


def test_writes_offline_html_and_uses_exact_stored_source_bundle(tmp_path,comparison):
    p=write_scenario_comparison(comparison,tmp_path/'comparison.html',
                                title='Two independently verified M19e scenarios')
    html=p.read_text()
    assert html.startswith('<!doctype html>')
    assert html.count('<iframe')==2
    assert 'srcdoc=' in html
    assert '<script id="comparison-data"' in html
    assert 'source identities' in html.lower()
    assert 'field:external' in html
    assert 'ring:A' in html and 'ring:B' in html
    assert 'no interpolation' in html.lower()
    assert 'gauge' in html.lower()
    assert 'http://' not in html and 'https://' not in html
    assert len(html)>200_000
    assert render_scenario_comparison_html(comparison,title='Two independently verified M19e scenarios')==html


def test_fails_closed_if_record_mutated_after_construction(comparison):
    c=deepcopy(comparison)
    c['changes'][0]['after']=[123,0,0]
    with pytest.raises(ValueError,match='altered'):
        render_scenario_comparison_html(c)


def test_special_characters_in_context_are_safely_escaped(comparison):
    c=compare_prescribed_scenarios(*comparison['scenario_bundles'],
        labels=('Original <scenario>', 'Altered & verified'),
        source_context='Compare <unsafe> & physically known models')
    html=render_scenario_comparison_html(c,title='Scenario <check> & other')
    assert 'Scenario &lt;check&gt; &amp; other' in html
    assert 'Scenario <check>' not in html
    assert 'Compare &lt;unsafe&gt; &amp; physically known' in html
    assert 'Original &lt;scenario&gt;' in html


def test_independent_uniform_field_contribution_at_same_original_launch(comparison):
    """Known uniform-field delta is independently determined by F=qE, not rendering."""
    left,right=(b['prescribed_motion'] for b in comparison['scenario_bundles'])
    a,b=left['samples'][0],right['samples'][0]
    q=left['particle']['charge_c']
    mass=left['mass_kg']
    assert a['position_m']==b['position_m']
    assert q==right['particle']['charge_c']
    assert b['electric_field_v_m'][0]-a['electric_field_v_m'][0]==pytest.approx(100000.)
    assert b['force_n'][0]-a['force_n'][0]==pytest.approx(q*100000.)
    assert b['acceleration_m_s2'][0]-a['acceleration_m_s2'][0]==pytest.approx(q*100000./mass)
    assert b['force_n'][1]-a['force_n'][1]==pytest.approx(0.,abs=1e-25)
    assert b['force_n'][2]-a['force_n'][2]==pytest.approx(0.,abs=1e-25)
