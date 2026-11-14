"""Physical revisions are independent models, not modified stored trajectories."""
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from conductor_sim import (
    EPSILON_0, PointCharge, PrescribedField, TestChargeMotionSpec as Spec,
    UniformElectricField, FixedPointField, UniformRingAxisField,
    UniformDiskAxisField, UniformRodField, UniformSphereField, UniformShellField,
    UnsupportedProblem, integrate_test_charge, prepare_scenario_comparison,
)
from conductor_sim.scenarios import (
    PrescribedScenario, revise_scenario, sample_scenario_viewer,
)


def uniform():
    spec = Spec(PrescribedField((UniformElectricField((2., 0., 0.)),)),
                PointCharge(2.), 4., (0., 0., 0.), .2, .02)
    return PrescribedScenario('base', spec, (('external', 'External field'),),
                              'Explicit uniform-field teaching model.')


def rings():
    spec = Spec(PrescribedField((
        UniformRingAxisField(1e-9, .3, axis=(1., 0., 0.)),
        UniformRingAxisField(-2e-9, .3, center_m=(.4, 0., 0.), axis=(1., 0., 0.)),
        UniformElectricField((2., 0., 0.)),
    )), PointCharge(1e-6), .1, (0., 0., 0.), .1, .02)
    return PrescribedScenario('rings', spec,
        (('ring:A', 'Ring A'), ('ring:B', 'Ring B'), ('external', 'External field')),
        'Explicit ring-axis teaching model.')


def revise(base, **kwargs):
    return revise_scenario(base, scenario_id='variant', note='Teacher trial.', **kwargs)


def test_immutable_parent_and_detached_inputs_and_audit():
    base = rings()
    before = base.as_dict()
    patch = {'ring:B': {'center_m': [.5, 0., 0.]}}
    variant = revise(base, source_changes=patch)
    patch['ring:B']['center_m'][0] = 999.
    assert variant.spec.field.sources[1].center_m == (.5, 0., 0.)
    audit = variant.as_dict()
    audit['spec']['field']['sources'][1]['center_m'][0] = 888.
    assert variant.spec.field.sources[1].center_m == (.5, 0., 0.)
    assert base.as_dict() == before
    assert variant.source_annotations == base.source_annotations
    with pytest.raises(FrozenInstanceError):
        variant.spec.mass_kg = 7.
    with pytest.raises(FrozenInstanceError):
        variant.changes[0].variation = 7.


def test_annotations_are_copied_not_retained():
    original = uniform()
    annotations = [['external', 'External field']]
    base = PrescribedScenario('base', original.spec, annotations, 'Authored model.')
    annotations[0][0] = 'changed'
    assert base.source_annotations == (('external', 'External field'),)


def test_lineage_fingerprint_and_provenance():
    original = uniform()
    base = PrescribedScenario('given', original.spec, original.source_annotations,
                              'Author-supplied problem data.', origin='source-given')
    variant = revise(base, motion_changes={'mass_kg': 8.})
    assert variant.origin == 'teacher-hypothetical'
    assert variant.viewer_provenance.startswith('Teacher hypothetical:')
    assert variant.parent_revision_id == base.revision_id
    assert len(variant.revision_id) == 64
    assert variant.revision_id != base.revision_id
    assert variant.as_dict()['revision_id'] == variant.revision_id
    assert revise(base, motion_changes={'mass_kg': 8.}).revision_id == variant.revision_id
    child = revise_scenario(variant, scenario_id='child', note='Reverse probe charge.',
                            particle_changes={'charge_c': -2.})
    assert child.parent_revision_id == variant.revision_id
    assert base.spec.mass_kg == 4. and variant.spec.mass_kg == 8.


def test_actual_changes_not_narrative_and_rebuild_classification():
    base = rings()
    variant = revise_scenario(base, scenario_id='revision', note='A deliberately misleading note.',
                               source_changes={'ring:B': {'charge_c': -3e-9}},
                               motion_changes={'mass_kg': .2})
    assert [(c.component_id, c.property) for c in variant.changes] == [
        ('ring:B', 'charge_c'), ('test-particle', 'mass_kg')]
    assert variant.required_action == 'reassemble-field-and-reintegrate'
    assert revise(base, motion_changes={'mass_kg': .2}).required_action == 'reintegrate-motion'
    assert base.required_action == 'validate-and-integrate'
    # Numerical controls, source order/types and the analysis plane stay fixed.
    for name in ('duration_s', 'sample_interval_s', 'plane_normal',
                 'relative_tolerance', 'length_scale_m', 'source_guard_m', 'max_internal_steps'):
        assert getattr(base.spec, name) == getattr(variant.spec, name)


def test_ids_not_source_indices_or_labels():
    base = rings()
    reordered = PrescribedScenario('reordered',
        replace(base.spec, field=PrescribedField(tuple(reversed(base.spec.field.sources)))),
        tuple(reversed(base.source_annotations)), 'Same sources in another authored order.')
    variant = revise(reordered, source_changes={'ring:A': {'charge_c': 4e-9}})
    assert variant.spec.field.sources[2].charge_c == 4e-9
    assert variant.spec.field.sources[1] == reordered.spec.field.sources[1]
    with pytest.raises(ValueError, match='unknown stable'):
        revise(base, source_changes={'Ring A': {'charge_c': 4e-9}})


def test_coupled_changes_are_validated_atomically():
    base = rings()
    before = base.as_dict()
    patches = {'ring:A': {'center_m': (0., .5, 0.)},
               'ring:B': {'center_m': (.4, .5, 0.)}}
    with pytest.raises(UnsupportedProblem):
        revise(base, source_changes=patches)
    variant = revise(base, source_changes=patches,
                     particle_changes={'position_m': (0., .5, 0.)})
    assert variant.spec.particle.position_m == (0., .5, 0.)
    assert base.as_dict() == before


@pytest.mark.parametrize('kwargs', [
    {'source_changes': {'missing': {'charge_c': 2e-9}}},
    {'source_changes': {'ring:A': {'kind': 'UniformShellField'}}},
    {'source_changes': {'ring:A': {'id': 'new'}}},
    {'source_changes': {'ring:A': {'radius_m': -1.}}},
    {'source_changes': {'ring:A': {'charge_c': 0.}}},
    {'source_changes': {'ring:A': {'radius_m': float('nan')}}},
    {'source_changes': {'ring:A': {'charge_c': True}}},
    {'particle_changes': {'charge_c': 0.}},
    {'particle_changes': {'position_m': (True, 0., 0.)}},
    {'particle_changes': {'position_m': (0., 0.)}},
    {'motion_changes': {'mass_kg': 0.}},
    {'motion_changes': {'mass_kg': float('inf')}},
    {'motion_changes': {'initial_velocity_m_s': (3e7, 0., 0.)}},
    {'motion_changes': {'duration_s': 2.}},
    {'motion_changes': {'relative_tolerance': .01}},
    {'motion_changes': {'plane_normal': (0., 0., 1.)}},
    {'motion_changes': {'source_guard_m': 0.}},
    {'medium_changes': {'permittivity_f_m': 0.}},
    {'medium_changes': {'temperature': 300.}},
])
def test_invalid_patches_leave_parent_intact(kwargs):
    base = rings()
    before = base.as_dict()
    with pytest.raises((ValueError, TypeError, UnsupportedProblem)):
        revise(base, **kwargs)
    assert base.as_dict() == before


@pytest.mark.parametrize('kwargs', [
    {'source_changes': {'external': {'field_v_m': (0., 2., 0.)}}},
    {'source_changes': {'ring:B': {'center_m': (.4, .1, 0.)}}},
    {'source_changes': {'ring:B': {'axis': (0., 0., 1.)}}},
    {'particle_changes': {'position_m': (0., .1, 0.)}},
    {'motion_changes': {'initial_velocity_m_s': (0., 0., .1)}},
])
def test_revised_axis_contract_is_rechecked(kwargs):
    with pytest.raises(UnsupportedProblem):
        revise(rings(), **kwargs)


@pytest.mark.parametrize('kwargs', [
    {}, {'motion_changes': {'mass_kg': .1}},
    {'source_changes': {'ring:A': {'axis': (2., 0., 0.)}}},
])
def test_noop_revisions_are_rejected(kwargs):
    with pytest.raises(ValueError, match='no actual'):
        revise(rings(), **kwargs)


@pytest.mark.parametrize('annotations', [
    None, (), (('external', 'A'), ('extra', 'B')),
    (('not a stable id', 'Field'),), (('external', ''),),
    (('test-particle', 'Field'),), (('medium', 'Field'),),
])
def test_invalid_identity_binding(annotations):
    with pytest.raises(ValueError):
        PrescribedScenario('base', uniform().spec, annotations, 'Authored model.')


def test_duplicate_ids_and_labels_rejected():
    base = rings()
    for annotations in (
        (('a', 'A'), ('a', 'B'), ('c', 'C')),
        (('a', 'A'), ('b', 'A'), ('c', 'C')),
    ):
        with pytest.raises(ValueError):
            PrescribedScenario('base', base.spec, annotations, 'Authored model.')


@pytest.mark.parametrize('bad', [[], 'mass_kg', 3])
def test_malformed_patch_containers(bad):
    with pytest.raises(TypeError):
        revise(uniform(), motion_changes=bad)


def test_distinct_scenario_ids_and_bounded_notes():
    base = uniform()
    for identity, note in [('base', 'Valid note.'), ('bad id', 'Valid note.'), ('new', '')]:
        with pytest.raises(ValueError):
            revise_scenario(base, scenario_id=identity, note=note, motion_changes={'mass_kg': 8.})


@pytest.mark.parametrize('source,patch', [
    (UniformElectricField((2., 0., 0.)), {'field_v_m': (3., 0., 0.)}),
    (FixedPointField(1e-9), {'position_m': (.2, 0., 0.)}),
    (UniformRingAxisField(1e-9, .3, axis=(1., 0., 0.)), {'radius_m': .4}),
    (UniformDiskAxisField(1e-9, .3, axis=(1., 0., 0.)), {'charge_c': 2e-9}),
    (UniformRodField(1e-9, (0., 0., -.5), (0., 0., .5)), {'charge_c': 2e-9}),
    (UniformSphereField(1e-9, .3), {'center_m': (.2, 0., 0.)}),
    (UniformShellField(1e-9, .3), {'radius_m': .4}),
])
def test_all_existing_source_families(source, patch):
    spec = Spec(PrescribedField((source,)), PointCharge(1e-9, (2., 0., 0.)),
                1., (0., 0., 0.), .1, .02)
    base = PrescribedScenario('base', spec, (('source', 'Source'),), 'Teaching source.')
    variant = revise(base, source_changes={'source': patch})
    assert type(variant.spec.field.sources[0]) is type(source)
    assert len(variant.changes) == 1
    assert integrate_test_charge(variant.spec).termination == 'duration'


def test_changed_geometry_rechecks_initial_guard():
    base = PrescribedScenario('point',
        Spec(PrescribedField((FixedPointField(1e-9),)), PointCharge(1e-9, (1., 0., 0.)),
             1., (0., 0., 0.), .1, .02), (('source', 'Point'),), 'Teaching model.')
    with pytest.raises(ValueError, match='guard'):
        revise(base, source_changes={'source': {'position_m': (1., 0., 0.)}})


def test_uniform_mass_revision_against_independent_constant_acceleration():
    base = uniform()
    variant = revise(base, motion_changes={'mass_kg': 8.})
    for scenario in (base, variant):
        result = integrate_test_charge(scenario.spec)
        acceleration = 2. * 2. / scenario.spec.mass_kg
        for row in result.samples:
            assert row.position_m[0] == pytest.approx(.5 * acceleration * row.time_s**2, abs=1e-12)
            assert row.velocity_m_s[0] == pytest.approx(acceleration * row.time_s, abs=1e-12)
            assert row.force_n == pytest.approx((4., 0., 0.))
            assert row.total_energy_j == pytest.approx(0., abs=1e-12)


def test_real_initial_conditions_and_charge_reversal():
    base = uniform()
    variant = revise(base, particle_changes={'position_m': (.3, 0., 0.), 'charge_c': -2.},
                     motion_changes={'initial_velocity_m_s': (1., 0., 0.)})
    result = integrate_test_charge(variant.spec)
    assert result.samples[0].position_m == (.3, 0., 0.)
    assert result.samples[0].velocity_m_s == (1., 0., 0.)
    assert result.samples[-1].position_m[0] == pytest.approx(.3 + .2 - .5 * .2**2)
    assert result.samples[-1].velocity_m_s[0] == pytest.approx(.8)


def test_gauge_revision_reintegrated_with_unchanged_motion():
    base = uniform()
    variant = revise(base, source_changes={'external': {'reference_potential_v': 17.}})
    left, right = integrate_test_charge(base.spec), integrate_test_charge(variant.spec)
    for a, b in zip(left.samples, right.samples):
        assert a.position_m == pytest.approx(b.position_m, abs=1e-13)
        assert a.velocity_m_s == pytest.approx(b.velocity_m_s, abs=1e-13)
        assert a.force_n == b.force_n
        assert b.potential_energy_j - a.potential_energy_j == pytest.approx(34.)


def test_medium_revision_has_expected_field_scaling():
    original = rings()
    base = PrescribedScenario('point',
        replace(original.spec, field=PrescribedField((FixedPointField(1e-9, (1., 0., 0.)),))),
        (('source', 'Point'),), 'Teaching model.')
    variant = revise(base, medium_changes={'permittivity_f_m': 2 * EPSILON_0})
    assert variant.spec.field.electric_field_at((0., 0., 0.))[0] == pytest.approx(
        .5 * base.spec.field.electric_field_at((0., 0., 0.))[0])


def test_fresh_solves_and_existing_comparison_ledger(monkeypatch):
    import conductor_sim.prescribed_viewer as viewer
    original = viewer.integrate_test_charge
    calls = []
    def record(spec):
        calls.append(spec)
        return original(spec)
    monkeypatch.setattr(viewer, 'integrate_test_charge', record)
    base = rings()
    variant = revise(base, source_changes={'ring:B': {'center_m': (.5, 0., 0.)}})
    options = dict(visual_frames=3, profile_samples=21, resolution=9,
                   extent_m=2., axis_orientation='horizontal')
    left, right = (sample_scenario_viewer(s, **options) for s in (base, variant))
    manifest = prepare_scenario_comparison(left, right,
        baseline_id=base.scenario_id, variation_id=variant.scenario_id,
        baseline_provenance=base.viewer_provenance, variation_provenance=variant.viewer_provenance)
    assert calls == [base.spec, variant.spec]
    assert manifest['revisions'] == json.loads(json.dumps([asdict(c) for c in variant.changes]))
    assert manifest['time_interpolation'] is False


def test_admissible_revision_need_not_have_a_comparable_viewer_plane():
    base = uniform()
    variant = revise(base, particle_changes={'position_m': (.5, 0., 0.)})
    left, right = (sample_scenario_viewer(s, visual_frames=3, resolution=9, extent_m=2.)
                   for s in (base, variant))
    with pytest.raises(ValueError, match='slice'):
        prepare_scenario_comparison(left, right)


def test_source_guard_run_is_not_a_complete_comparison():
    base = PrescribedScenario('guard',
        Spec(PrescribedField((FixedPointField(-1e-9),)), PointCharge(1e-9, (.1, 0., 0.)),
             1., (-1., 0., 0.), .2, .02, source_guard_m=.01),
        (('source', 'Point'),), 'Intentional guard approach.')
    variant = revise(base, motion_changes={'mass_kg': 2.})
    left, right = (sample_scenario_viewer(s, visual_frames=3, resolution=9, extent_m=2.)
                   for s in (base, variant))
    assert left['prescribed_motion']['termination'] == 'source-guard'
    assert right['prescribed_motion']['termination'] == 'source-guard'
    with pytest.raises(ValueError):
        prepare_scenario_comparison(left, right)


@pytest.mark.parametrize('bad', [None, [], 'charge_c', 3])
def test_malformed_nested_source_patches(bad):
    with pytest.raises(TypeError):
        revise(rings(), source_changes={'ring:A': bad})


@pytest.mark.parametrize('case', [
    'double-ring-a-charge', 'ring-b-shift', 'double-mass',
    'reverse-field', 'launch-speed', 'gauge-shift',
])
def test_six_example_revisions_match_computed_comparison(case):
    from examples.revise_prescribed_scenarios import make_scenarios
    base, variant = make_scenarios(case)
    options = dict(visual_frames=3, profile_samples=21, resolution=9,
                   extent_m=2.25, axis_orientation='horizontal')
    left, right = (sample_scenario_viewer(s, **options) for s in (base, variant))
    manifest = prepare_scenario_comparison(left, right)
    assert manifest['revisions'] == json.loads(json.dumps([asdict(c) for c in variant.changes]))
    assert len(manifest['revisions']) == 1
