"""M22a: atomic structural authoring over existing qualified prescribed models.

This module assembles immutable components and provenance, never fields,
forces, trajectories, observations or comparison policies. Existing parameter
revisions and their serialization are deliberately unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from hashlib import sha256
import json

from .conservative_forces import UniformGravity
from .prescribed_fields import PrescribedField, SOURCE_TYPES
from .prescribed_viewer import _source_annotations
from .scenarios import PrescribedScenario, _text

_UNCHANGED = object()
_RESERVED = frozenset(("test-particle", "medium", "gravity"))
_MAX_DEPTH = 16


@dataclass(frozen=True, slots=True)
class NamedSource:
    """Explicit source identity, display label and existing built-in component."""
    source_id: str
    label: str
    source: object

    def __post_init__(self):
        if type(self.source) not in SOURCE_TYPES:
            raise TypeError("structural authoring requires a supported built-in fixed source")
        # Share the current identity/label contract; never normalize a source
        # by reconstructing it (which could renormalize a tilted axis).
        field = PrescribedField((self.source,))
        _source_annotations(type("_Binding", (), {"field": field})(),
                            ((self.source_id, self.label),))
        if self.source_id in _RESERVED:
            raise ValueError("structural source identity is reserved for another physical role")

    def as_dict(self):
        return dict(id=self.source_id, label=self.label, source=self.source.as_dict())


@dataclass(frozen=True, slots=True)
class SourceReplacement:
    """Retire one source identity and introduce another at its inventory slot."""
    old_id: str
    new: NamedSource

    def __post_init__(self):
        if type(self.old_id) is not str or not self.old_id:
            raise ValueError("replacement needs the old stable source identity")
        if type(self.new) is not NamedSource:
            raise TypeError("replacement requires a NamedSource")
        if self.old_id == self.new.source_id:
            raise ValueError("replacement requires a NEW source identity; use revise_scenario for properties")


def _sequence(value, cls, name):
    if type(value) not in (tuple, list) or any(type(x) is not cls for x in value):
        raise TypeError(f"{name} must be a sequence of {cls.__name__}")
    return tuple(value)


def _model(parent):
    if type(parent) is PrescribedScenario:
        return parent
    if type(parent) is ScenarioReassembly:
        return parent.scenario
    raise TypeError("parent must be PrescribedScenario or ScenarioReassembly")


def _sources(model):
    return tuple(NamedSource(identity, label, source)
                 for (identity, label), source in zip(
                     model.source_annotations, model.spec.field.sources))


def _history(parent):
    """Return supplied ancestry only; this is not a global identity registry."""
    scenarios, source_ids, depth = set(), set(), 0
    while True:
        model = _model(parent)
        scenarios.add(model.scenario_id)
        source_ids.update(identity for identity, _ in model.source_annotations)
        if type(parent) is PrescribedScenario:
            return scenarios, source_ids, depth
        depth += 1
        if depth >= _MAX_DEPTH:
            raise ValueError("structural ancestry exceeds the bounded 16-step record")
        parent = parent.parent


@dataclass(frozen=True, slots=True)
class ScenarioReassembly:
    """Bound immediate models plus retained structural ancestry.

    ``scenario`` is an ordinary, freshly authored PrescribedScenario. Its old
    parameter-revision schema is not overloaded with component membership.
    Retain this record for structural lineage; pass .scenario to the existing
    integrate/view/parameter-revision interfaces. Digests identify records,
    not a trusted importer, proof of original-source provenance or a run.
    """
    parent: PrescribedScenario | ScenarioReassembly
    scenario: PrescribedScenario
    replacements: tuple[SourceReplacement, ...] = ()

    def __post_init__(self):
        old = _model(self.parent)
        ids, past, _ = _history(self.parent)
        if type(self.scenario) is not PrescribedScenario:
            raise TypeError("structural result needs an ordinary PrescribedScenario")
        new = self.scenario
        if new.scenario_id in ids:
            raise ValueError("structural child requires a new scenario identity")
        if new.origin != "teacher-hypothetical" or new.parent_revision_id is not None or new.changes:
            raise ValueError("structural child must be a fresh teacher-hypothetical model")
        replacements = _sequence(self.replacements, SourceReplacement, "replacements")
        object.__setattr__(self, "replacements", replacements)
        a, b = _sources(old), _sources(new)
        amap = {s.source_id: s for s in a}
        bmap = {s.source_id: s for s in b}
        for identity in amap.keys() & bmap.keys():
            if amap[identity] != bmap[identity]:
                raise ValueError("retained identities cannot change properties in a structural operation")
        if (bmap.keys() - amap.keys()) & past:
            raise ValueError("retired source identities cannot be reused within retained ancestry")
        targets = [r.old_id for r in replacements]
        fresh = [r.new.source_id for r in replacements]
        if len(set(targets)) != len(targets) or len(set(fresh)) != len(fresh):
            raise ValueError("replacement targets and new identities must be unique")
        for r in replacements:
            if r.old_id not in amap or r.old_id in bmap or bmap.get(r.new.source_id) != r.new:
                raise ValueError("replacement ledger disagrees with actual source membership")
        # Preserve analysis, initial conditions, medium and numerical controls.
        for f in fields(old.spec):
            if f.name not in ("field", "gravity") and getattr(old.spec, f.name) != getattr(new.spec, f.name):
                raise ValueError("structural operation cannot edit particle, analysis or numerical controls")
        if old.spec.field.permittivity_f_m != new.spec.field.permittivity_f_m:
            raise ValueError("structural operation cannot change medium properties")
        if old.spec.gravity is not None and new.spec.gravity is not None and old.spec.gravity != new.spec.gravity:
            raise ValueError("use revise_scenario for existing gravity properties")
        replacement_map = {r.old_id: r.new for r in replacements}
        prefix = tuple(replacement_map.get(s.source_id, s) for s in a
                       if s.source_id in bmap or s.source_id in replacement_map)
        if b[:len(prefix)] != prefix:
            raise ValueError("retained/replaced inventory slots must preserve original order")
        if not self.changes:
            raise ValueError("reassembly has no actual structural change")

    @property
    def required_action(self):
        return "independently-integrate-qualified-new-assembly"

    @property
    def changes(self):
        old, new = _model(self.parent), self.scenario
        a, b = _sources(old), _sources(new)
        amap, bmap = ({s.source_id: s for s in group} for group in (a, b))
        replacement_map = {r.old_id: r.new for r in self.replacements}
        replacements_new = {r.new.source_id for r in self.replacements}
        result = []
        for s in a:
            if s.source_id in replacement_map:
                result.append(dict(operation="replace-source", before=s.as_dict(),
                                   after=replacement_map[s.source_id].as_dict()))
            elif s.source_id not in bmap:
                result.append(dict(operation="remove-source", before=s.as_dict(), after=None))
        for s in b:
            if s.source_id not in amap and s.source_id not in replacements_new:
                result.append(dict(operation="add-source", before=None, after=s.as_dict()))
        if (old.spec.gravity is None) != (new.spec.gravity is None):
            result.append(dict(operation="add-gravity" if new.spec.gravity else "remove-gravity",
                               component_id="gravity",
                               before=old.spec.gravity.as_dict() if old.spec.gravity else None,
                               after=new.spec.gravity.as_dict() if new.spec.gravity else None))
        # Detached records: changing a consumer's dict cannot alter the model.
        return json.loads(json.dumps(result, allow_nan=False))

    def _record(self):
        old = _model(self.parent)
        return dict(
            schema="conductor_sim.scenario_reassembly.v1",
            origin="teacher-hypothetical", note=self.scenario.provenance,
            parent_revision_id=old.revision_id,
            child_revision_id=self.scenario.revision_id,
            parent_model=old.as_dict(), child_model=self.scenario.as_dict(),
            parent_assembly=(self.parent.as_dict()
                             if type(self.parent) is ScenarioReassembly else None),
            changes=self.changes, required_action=self.required_action,
            qualification="existing complete-model constructors accepted; not a numerical run",
        )

    @property
    def revision_id(self):
        record = json.dumps(self._record(), sort_keys=True,
                            separators=(",", ":"), allow_nan=False)
        return sha256(record.encode("utf-8")).hexdigest()

    def as_dict(self):
        record = self._record()
        record["revision_id"] = sha256(json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        return json.loads(json.dumps(record, allow_nan=False))


def reassemble_scenario(parent, *, scenario_id: str, note: str,
                        add_sources=(), remove_sources=(), replace_sources=(),
                        gravity=_UNCHANGED) -> ScenarioReassembly:
    """Assemble all structural edits, then reuse whole-system qualification.

    An omitted gravity argument preserves membership; UniformGravity adds a
    previously absent component; None removes it. Existing gravity properties
    belong to revise_scenario. A replacement always retires its old source ID.

    No state, backend, physical plane, duration, initial condition, tolerance
    or observation policy is inferred or relaxed. Invalid edits are atomic.
    """
    old = _model(parent)
    ids, past, _ = _history(parent)
    _text(note, "structural revision note", 400)
    if scenario_id in ids:
        raise ValueError("structural child requires a new scenario identity")
    additions = _sequence(add_sources, NamedSource, "add_sources")
    removals = _sequence(remove_sources, str, "remove_sources")
    replacements = _sequence(replace_sources, SourceReplacement, "replace_sources")
    bound = _sources(old)
    existing = {s.source_id: s for s in bound}
    targets = removals + tuple(r.old_id for r in replacements)
    if len(set(targets)) != len(targets):
        raise ValueError("a source cannot be removed/replaced more than once")
    if any(identity not in existing for identity in targets):
        raise ValueError("structural edit references an unknown stable source identity")
    incoming = additions + tuple(r.new for r in replacements)
    new_ids = [s.source_id for s in incoming]
    if len(set(new_ids)) != len(new_ids) or set(new_ids) & past:
        raise ValueError("new source identities must be unique and never reuse retained ancestry")
    replacement_map = {r.old_id: r.new for r in replacements}
    removed = set(removals)
    final = tuple(replacement_map.get(s.source_id, s) for s in bound
                  if s.source_id not in removed) + additions
    if gravity is _UNCHANGED:
        new_gravity = old.spec.gravity
    else:
        if gravity is not None and type(gravity) is not UniformGravity:
            raise TypeError("gravity must be a built-in UniformGravity or explicit None")
        if gravity is not None and old.spec.gravity is not None and gravity != old.spec.gravity:
            raise ValueError("use revise_scenario for existing gravity properties")
        new_gravity = gravity
    # One constructor call for the FINAL inventory. Never validate transient
    # half-edited mirror pairs; never project away unsupported forces.
    spec = replace(old.spec,
                   field=PrescribedField(tuple(s.source for s in final),
                                         old.spec.field.permittivity_f_m),
                   gravity=new_gravity)
    child = PrescribedScenario(
        scenario_id, spec, tuple((s.source_id, s.label) for s in final),
        note, origin="teacher-hypothetical",
    )
    return ScenarioReassembly(parent, child, replacements)
