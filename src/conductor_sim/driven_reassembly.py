"""M26a: opt-in structural changes to retained driven physical models.

This is a composition/provenance bridge, not a new force provider, integrator,
event search, renderer or trusted import path. Existing M22a whole-source
qualification and M24b per-schedule-phase qualification remain authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .driven_motion import DrivenModel
from .scenarios import _text
from .structural_scenarios import (
    ScenarioReassembly, _UNCHANGED, reassemble_scenario,
)

_MAX_DEPTH = 16


def _model(parent):
    if type(parent) is DrivenModel:
        return parent
    if type(parent) is DrivenReassembly:
        return parent.model
    raise TypeError("parent must be DrivenModel or DrivenReassembly")


def _driven_ids(parent):
    """Only the explicitly retained driven ancestry is an identity namespace."""
    identities = set()
    steps = 0
    while True:
        model = _model(parent)
        if model.model_id in identities:
            raise ValueError("repeated driven model identity in retained ancestry")
        identities.add(model.model_id)
        if type(parent) is DrivenModel:
            return identities
        steps += 1
        if steps >= _MAX_DEPTH:
            raise ValueError("driven structural ancestry exceeds 16 steps")
        parent = parent.parent


@dataclass(frozen=True, slots=True)
class DrivenReassembly:
    """Retain an exact parent, new qualified driven model and base edit ledger.

    `model` is an ordinary DrivenModel accepted by every M24b phase check.
    `base_assembly` is the actual M22a structural edit, preserving its own
    retired-ID ancestry. This record does *not* certify numerical completion,
    original PDF provenance, authenticated JSON import or physical contact.
    """

    parent: DrivenModel | DrivenReassembly
    model: DrivenModel
    base_assembly: ScenarioReassembly

    def __post_init__(self):
        old = _model(self.parent)
        if type(self.model) is not DrivenModel or type(self.base_assembly) is not ScenarioReassembly:
            raise TypeError("driven structural result requires built-in model and M22a ledger")
        new = self.model
        if new.model_id in _driven_ids(self.parent):
            raise ValueError("driven structural child needs a fresh model identity")
        if new.parent_revision_id != old.revision_id:
            raise ValueError("driven structural parent revision does not match")
        # A change to a force schedule or drag coefficient belongs to the
        # existing separate parameter-revision operation, never a hidden edit.
        if new.drag is not old.drag or new.applied is not old.applied:
            raise ValueError("structural reassembly must retain exact drag and applied schedule")
        if (new.base is not self.base_assembly.scenario or
                new.provenance != self.base_assembly.scenario.provenance):
            raise ValueError("structural child must bind the actual qualified base and note")
        expected_parent = (self.parent.base_assembly
                           if type(self.parent) is DrivenReassembly else old.base)
        if self.base_assembly.parent is not expected_parent:
            raise ValueError("base structural ledger does not follow the retained parent")
        if new.base.origin != "teacher-hypothetical":
            raise ValueError("structural child must be a teacher hypothetical")
        # M22a already verifies unchanged particle, launch, physical plane,
        # medium and numerical policy, and prohibits no-op membership changes.
        # M24b already verifies every applied phase, including late switches.

    @property
    def required_action(self):
        return "independently-integrate-qualified-new-driven-assembly"

    @property
    def changes(self):
        """Detached actual base-membership changes, not an author narrative."""
        return self.base_assembly.changes

    def _record(self):
        original = _model(self.parent)
        return dict(
            schema="conductor_sim.driven_reassembly.v1",
            origin="teacher-hypothetical",
            parent_model_revision_id=original.revision_id,
            child_model_revision_id=self.model.revision_id,
            parent_driven_assembly_revision_id=(
                self.parent.revision_id if type(self.parent) is DrivenReassembly else None
            ),
            parent_model=original.as_dict(),
            child_model=self.model.as_dict(),
            base_assembly=self.base_assembly.as_dict(),
            changes=self.changes,
            required_action=self.required_action,
            qualification=(
                "Complete M22a base construction and every M24b field phase "
                "accepted; no trajectory has been computed by reassembly."
            ),
        )

    @property
    def revision_id(self):
        payload = json.dumps(self._record(), sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self):
        """Detached JSON-native lineage; SHA-256 is identity, not authenticity."""
        record = self._record()
        record["revision_id"] = self.revision_id
        return json.loads(json.dumps(record, allow_nan=False))


def reassemble_driven_model(
    parent, *, model_id: str, scenario_id: str, note: str,
    add_sources=(), remove_sources=(), replace_sources=(), gravity=_UNCHANGED,
) -> DrivenReassembly:
    """Atomically structurally edit a driven model, preserving drag and schedule.

    Stable source edits use M22a and its full final-inventory qualification.
    Then the *entire* M24b model is requalified over all scheduled field phases.
    No particle initial state, axis/plane, drag or field schedule is inferred.
    Call investigate_driven_motion with a newly authored DrivenStudy to solve.
    """
    old = _model(parent)
    _text(note, "driven structural revision note", 400)
    if model_id in _driven_ids(parent):
        raise ValueError("driven structural child needs a fresh model identity")
    base_parent = (parent.base_assembly
                   if type(parent) is DrivenReassembly else old.base)
    edits = dict(scenario_id=scenario_id, note=note, add_sources=add_sources,
                 remove_sources=remove_sources, replace_sources=replace_sources)
    if gravity is not _UNCHANGED:
        edits["gravity"] = gravity
    base = reassemble_scenario(base_parent, **edits)
    # No partial child or result escapes if a later applied phase breaks the
    # axial/planar certificate. This constructor checks *every* phase.
    new = DrivenModel(base.scenario, old.drag, old.applied, model_id, note,
                      parent_revision_id=old.revision_id)
    return DrivenReassembly(parent, new, base)
