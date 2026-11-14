"""M20b: immutable, identity-addressed revisions of prescribed-field models.

A revision is an admissible authored model, not a solved/verified trajectory.
No field law, integrator, browser physics or persistent registry lives here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
import json
import re

from .model import PointCharge
from .prescribed_fields import PrescribedField, _number, _vector
from .prescribed_motion import TestChargeMotionSpec
from .prescribed_viewer import _source_annotations, sample_test_charge_viewer

_ID = re.compile(r'[A-Za-z][A-Za-z0-9:_-]{0,63}\Z')
_RESERVED = frozenset(('test-particle', 'medium'))


def _text(value, name, limit):
    if (type(value) is not str or not 1 <= len(value) <= limit or
            value != value.strip() or not value.isprintable()):
        raise ValueError(f'{name} must be trimmed, printable text (1 to {limit} characters)')
    return value


@dataclass(frozen=True, slots=True)
class ScenarioChange:
    """An actual normalized parameter difference, not a teacher's narrative."""
    component_id: str
    property: str
    baseline: float | tuple
    variation: float | tuple


@dataclass(frozen=True, slots=True)
class PrescribedScenario:
    """One reusable authored model and its explicit source identity binding.

    ``origin='source-given'`` is an author assertion, not PDF verification.
    Derived scenarios always have origin ``teacher-hypothetical``. The digest
    identifies this exact authored record, not a software version or proof of
    numerical correctness. No mutable dictionaries are retained.
    """
    scenario_id: str
    spec: TestChargeMotionSpec
    source_annotations: tuple[tuple[str, str], ...]
    provenance: str
    origin: str = 'teacher-hypothetical'
    parent_revision_id: str | None = field(default=None, init=False)
    changes: tuple[ScenarioChange, ...] = field(default=(), init=False)

    def __post_init__(self):
        if type(self.scenario_id) is not str or _ID.fullmatch(self.scenario_id) is None:
            raise ValueError('scenario_id must be a bounded stable ASCII identifier')
        if type(self.spec) is not TestChargeMotionSpec:
            raise TypeError('scenario requires an existing TestChargeMotionSpec')
        # Revision reconstructs a built-in field. Accepting custom subclasses
        # here could silently discard their behavior without a ledger entry.
        if (type(self.spec.field) is not PrescribedField or
                type(self.spec.particle) is not PointCharge):
            raise TypeError('scenario requires built-in PrescribedField and PointCharge, not subclasses')
        _text(self.provenance, 'provenance', 400)
        if self.origin not in ('source-given', 'teacher-hypothetical'):
            raise ValueError('origin must be source-given or teacher-hypothetical')
        annotations = _source_annotations(self.spec, self.source_annotations)
        if annotations is None:
            raise ValueError('scenario requires explicit source identity bindings')
        if self.spec.gravity is not None and any(a['id']=='gravity' for a in annotations):
            raise ValueError('gravity is reserved for the active mechanical component')
        if any(a['id'] in _RESERVED for a in annotations):
            raise ValueError('source identities test-particle and medium are reserved')
        object.__setattr__(self, 'source_annotations',
                           tuple((a['id'], a['label']) for a in annotations))

    @property
    def required_action(self) -> str:
        if self.parent_revision_id is None:
            return 'validate-and-integrate'
        if any(c.component_id == 'gravity' for c in self.changes):
            return 'reassemble-forces-and-reintegrate'
        if any(c.component_id != 'test-particle' for c in self.changes):
            return 'reassemble-field-and-reintegrate'
        return 'reintegrate-motion'

    @property
    def viewer_provenance(self) -> str:
        prefix = ('Teacher hypothetical' if self.origin == 'teacher-hypothetical'
                  else 'Source-given model (author assertion)')
        return prefix + ': ' + self.provenance

    def _record(self):
        spec = asdict(self.spec)
        # Preserve source *types*, which dataclasses.asdict alone would erase.
        spec['field'] = self.spec.field.as_dict()
        if self.spec.gravity is None:
            spec.pop('gravity')  # Preserve all original electric-only hashes.
        else:
            spec['gravity'] = self.spec.gravity.as_dict()
        return dict(schema='conductor_sim.prescribed_scenario.v1',
                    scenario_id=self.scenario_id, origin=self.origin,
                    provenance=self.provenance, spec=spec,
                    source_annotations=self.source_annotations,
                    parent_revision_id=self.parent_revision_id,
                    changes=[asdict(c) for c in self.changes],
                    required_action=self.required_action)

    @property
    def revision_id(self) -> str:
        payload = json.dumps(self._record(), sort_keys=True,
                             separators=(',', ':'), allow_nan=False)
        return sha256(payload.encode('utf-8')).hexdigest()

    def as_dict(self) -> dict:
        """Detached JSON-native audit record; not a trusted import format."""
        record = self._record()
        record['revision_id'] = self.revision_id
        return json.loads(json.dumps(record, allow_nan=False))


def _mapping(value, name):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping')
    return dict(value)


def _patch(original, changes, allowed, name):
    changes = _mapping(changes, name)
    if any(type(k) is not str or k not in allowed for k in changes):
        raise ValueError(f'{name} contains an unsupported property; allowed: {sorted(allowed)}')
    normalized = {}
    for key, value in changes.items():
        before = getattr(original, key)
        normalized[key] = (_vector(value, key) if isinstance(before, tuple)
                           else _number(value, key))
    # Reuse the original dataclass constructor's physical validation.
    if all(getattr(original, k) == v for k, v in normalized.items()):
        return original
    return replace(original, **normalized)


def _differences(identity, old, new, names):
    return tuple(ScenarioChange(identity, name, getattr(old, name), getattr(new, name))
                 for name in sorted(names) if getattr(old, name) != getattr(new, name))


def revise_scenario(parent: PrescribedScenario, *, scenario_id: str, note: str,
                    source_changes: Mapping | None = None,
                    particle_changes: Mapping | None = None,
                    motion_changes: Mapping | None = None,
                    medium_changes: Mapping | None = None,
                    gravity_changes: Mapping | None = None) -> PrescribedScenario:
    """Atomically revise supported physical properties without mutating parent.

    Source patches are keyed by stable ID, never source index or display label.
    Particle: charge_c, position_m. Motion: mass_kg, initial_velocity_m_s.
    Gravity: acceleration_m_s2, reference_position_m on an existing component.
    Medium: permittivity_f_m. Source properties are the existing built-in
    dataclass's physical parameters. IDs, labels, kinds, source membership,
    duration, sampling, tolerances, guards and analysis plane cannot be patched.
    All coupled changes are assembled before the motion contract is rechecked.
    Unsupported geometry/launches raise the original validation exception.
    """
    if type(parent) is not PrescribedScenario:
        raise TypeError('parent must be PrescribedScenario')
    _text(note, 'revision note', 400)
    if scenario_id == parent.scenario_id:
        raise ValueError('a revision requires a distinct scenario_id')
    patches = _mapping(source_changes, 'source_changes')
    identities = {identity for identity, _ in parent.source_annotations}
    if any(type(identity) is not str or identity not in identities for identity in patches):
        raise ValueError('source_changes contains an unknown stable source identity')
    sources, changes = [], []
    for (identity, _), source in zip(parent.source_annotations, parent.spec.field.sources):
        names = {f.name for f in fields(source)}
        if identity in patches and not isinstance(patches[identity], Mapping):
            raise TypeError('each source patch must be a mapping of physical properties')
        revised = _patch(source, patches.get(identity), names, f'source {identity}')
        sources.append(revised)
        changes.extend(_differences(identity, source, revised, names))
    medium = _mapping(medium_changes, 'medium_changes')
    if any(type(k) is not str or k != 'permittivity_f_m' for k in medium):
        raise ValueError('medium_changes permits only permittivity_f_m')
    eps = (_number(medium['permittivity_f_m'], 'permittivity_f_m', positive=True)
           if 'permittivity_f_m' in medium else parent.spec.field.permittivity_f_m)
    new_field = PrescribedField(tuple(sources), permittivity_f_m=eps)
    changes.extend(_differences('medium', parent.spec.field, new_field, ('permittivity_f_m',)))
    particle = _patch(parent.spec.particle, particle_changes,
                      {'charge_c', 'position_m'}, 'particle_changes')
    changes.extend(_differences('test-particle', parent.spec.particle, particle,
                                ('charge_c', 'position_m')))
    motion = _mapping(motion_changes, 'motion_changes')
    allowed = {'mass_kg', 'initial_velocity_m_s'}
    if any(type(k) is not str or k not in allowed for k in motion):
        raise ValueError('motion_changes permits only mass_kg and initial_velocity_m_s; numerical controls are not physical revisions')
    motion = {k: (_vector(v, k) if k == 'initial_velocity_m_s' else _number(v, k, positive=True))
              for k, v in motion.items()}
    # Crucially, validate the *whole revised system*, not intermediate patches.
    gravity_patch = _mapping(gravity_changes, 'gravity_changes')
    gravity = parent.spec.gravity
    if gravity_patch and gravity is None:
        raise ValueError('gravity_changes requires an authored gravity component; membership is not a parameter patch')
    if gravity is not None:
        gravity = _patch(gravity, gravity_patch,
                         {'acceleration_m_s2', 'reference_position_m'}, 'gravity_changes')
    spec = replace(parent.spec, field=new_field, particle=particle, gravity=gravity, **motion)
    changes.extend(_differences('test-particle', parent.spec, spec, allowed))
    if gravity is not None:
        changes.extend(_differences('gravity', parent.spec.gravity, gravity,
                                    ('acceleration_m_s2', 'reference_position_m')))
    if not changes:
        raise ValueError('revision has no actual physical or potential-reference change')
    revision = PrescribedScenario(scenario_id, spec, parent.source_annotations,
                                  note, origin='teacher-hypothetical')
    object.__setattr__(revision, 'parent_revision_id', parent.revision_id)
    object.__setattr__(revision, 'changes', tuple(changes))
    return revision


def sample_scenario_viewer(scenario: PrescribedScenario, *, visual_frames=16,
                           resolution=17, profile_samples=201, extent_m=None,
                           axis_orientation='vertical') -> dict:
    """Independently integrate, then use the unchanged M19e validation/viewer.

    There is deliberately no cached-result argument. A source-guard outcome
    remains a censored run; the existing M20a comparison rejects it. Matching
    model validity does not guarantee matching comparison slices/time grids.
    """
    if type(scenario) is not PrescribedScenario:
        raise TypeError('scenario must be PrescribedScenario')
    return sample_test_charge_viewer(
        scenario.spec, source_annotations=scenario.source_annotations,
        visual_frames=visual_frames, resolution=resolution,
        profile_samples=profile_samples, extent_m=extent_m,
        axis_orientation=axis_orientation)
