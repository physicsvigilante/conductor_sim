"""M32b: typed native source-ID ancestry across mixed structural/physical graph edges.

The source identity ledger starts at the supplied, qualified graph ROOT.
M22a preserves source history while reassembling structural worlds but an
intermediate M20b property revision is a plain PrescribedScenario; the
certificate preserves already-used IDs across that otherwise lossy edge.

This is a branch-local in-process ancestry record, not a source registry,
a model importer, a field solver, or PDF-source authentication.
"""
from __future__ import annotations

from dataclasses import dataclass

from .scenarios import PrescribedScenario


def _string_tuple(value, name):
    if (type(value) is not tuple or
            any(type(item) is not str or not item for item in value)):
        raise TypeError(f"{name} must be an immutable tuple of native ID strings")
    return value


def _native_ids(model: PrescribedScenario):
    if type(model) is not PrescribedScenario:
        raise TypeError("source ancestry requires a qualified native model")
    ids = tuple(identity for identity, _ in model.source_annotations)
    if len(ids) != len(set(ids)):
        raise ValueError("one source-model inventory cannot reuse an identity")
    return ids


@dataclass(frozen=True, slots=True)
class SourceWorldAncestry:
    """A native runner-derived provenance certificate for ONE qualified world.

    All path/ID tuples hold exact built-in strings, not arbitrary objects or
    custom equality. A previous source ID cannot be newly introduced on its
    own ancestral path, even after one or more physical-property revisions.
    Other alternative branches keep their own histories.
    """

    world_id: str
    parent_world_id: str | None
    model_revision_id: str
    parent_revision_id: str | None
    world_path: tuple[str, ...]
    revision_path: tuple[str, ...]
    active_source_ids: tuple[str, ...]
    seen_source_ids: tuple[str, ...]
    retired_source_ids: tuple[str, ...]
    introduced_source_ids: tuple[str, ...]
    retired_in_this_step: tuple[str, ...]
    gravity_present: bool

    def __post_init__(self):
        for name in ("world_id", "model_revision_id"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise TypeError("source ancestry requires exact native world/revision strings")
        for name in ("parent_world_id", "parent_revision_id"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise TypeError("source ancestry parent identity must be a native string")
        for name in (
            "world_path", "revision_path", "active_source_ids", "seen_source_ids",
            "retired_source_ids", "introduced_source_ids", "retired_in_this_step",
        ):
            _string_tuple(getattr(self, name), name)
        if (not self.world_path or self.world_path[-1] != self.world_id or
                len(self.world_path) != len(self.revision_path) or
                self.revision_path[-1] != self.model_revision_id):
            raise ValueError("source ancestry world and revision paths disagree")
        if self.parent_world_id is None:
            if (self.world_id != "baseline" or self.parent_revision_id is not None or
                    len(self.world_path) != 1 or self.introduced_source_ids or
                    self.retired_source_ids or self.retired_in_this_step):
                raise ValueError("graph root ancestry cannot invent a prior history")
        elif (len(self.world_path) <= 1 or
              self.world_path[-2] != self.parent_world_id or
              self.revision_path[-2] != self.parent_revision_id):
            raise ValueError("source ancestry does not match immediate model parent")
        if (len(self.active_source_ids) != len(set(self.active_source_ids)) or
                len(self.seen_source_ids) != len(set(self.seen_source_ids)) or
                len(self.introduced_source_ids) != len(set(self.introduced_source_ids)) or
                any(i not in self.seen_source_ids for i in self.active_source_ids) or
                tuple(i for i in self.seen_source_ids if i not in self.active_source_ids)
                != self.retired_source_ids or
                any(i not in self.active_source_ids for i in self.introduced_source_ids) or
                any(i not in self.retired_source_ids for i in self.retired_in_this_step)):
            raise ValueError("source ancestry inventory/retirement ledger inconsistent")
        if type(self.gravity_present) is not bool:
            raise TypeError("gravity membership must be a native Boolean")

    def as_dict(self):
        return dict(
            schema="conductor_sim.source_world_ancestry.v1",
            world_id=self.world_id,
            parent_world_id=self.parent_world_id,
            model_revision_id=self.model_revision_id,
            parent_revision_id=self.parent_revision_id,
            world_path=self.world_path,
            revision_path=self.revision_path,
            active_source_ids=self.active_source_ids,
            seen_source_ids=self.seen_source_ids,
            retired_source_ids=self.retired_source_ids,
            introduced_source_ids=self.introduced_source_ids,
            retired_in_this_step=self.retired_in_this_step,
            gravity_present=self.gravity_present,
            history_scope=("source identities known from supplied graph root and "
                           "actually accepted descendants only; no global retired-ID registry"),
        )


def source_ancestry_root(model: PrescribedScenario) -> SourceWorldAncestry:
    """Certify the supplied graph root, without inventing older source history."""
    ids = _native_ids(model)
    return SourceWorldAncestry(
        "baseline", None, model.revision_id, None,
        ("baseline",), (model.revision_id,), ids, ids, (), (), (),
        model.spec.gravity is not None,
    )


def advance_source_ancestry(
    previous: SourceWorldAncestry,
    *, world_id: str,
    parent_id: str,
    child: PrescribedScenario,
    operation: str,
) -> SourceWorldAncestry:
    """Reconstruct ONE admitted edge from actual native models and previous cert.

    The caller checks a real M22a assembly or M20b revision before this call.
    In particular, an attempted reuse of a previously *retired* source ID is
    refused, even when the M22a direct parent is a plain M20b revised model.
    """
    if type(previous) is not SourceWorldAncestry or type(child) is not PrescribedScenario:
        raise TypeError("source ancestry extension needs authentic native types")
    if type(world_id) is not str or not world_id or type(parent_id) is not str:
        raise TypeError("ancestry edge requires exact native graph IDs")
    if parent_id != previous.world_id or world_id in previous.world_path:
        raise ValueError("ancestry edge lost real immediate graph parent")
    if operation not in ("structural", "physical") or type(operation) is not str:
        raise ValueError("ancestry edge must be a real structural or physical revision")
    active = _native_ids(child)
    prior_active = previous.active_source_ids
    introduced = tuple(i for i in active if i not in prior_active)
    if operation == "physical" and active != prior_active:
        raise ValueError("physical property revision cannot alter named source membership")
    if any(i in previous.seen_source_ids for i in introduced):
        raise ValueError("retired or existing source identity cannot be reintroduced in ancestry")
    seen = previous.seen_source_ids + introduced
    retired = tuple(i for i in seen if i not in active)
    retired_now = tuple(i for i in prior_active if i not in active)
    return SourceWorldAncestry(
        world_id=world_id, parent_world_id=parent_id,
        model_revision_id=child.revision_id,
        parent_revision_id=previous.model_revision_id,
        world_path=previous.world_path + (world_id,),
        revision_path=previous.revision_path + (child.revision_id,),
        active_source_ids=active,
        seen_source_ids=seen,
        retired_source_ids=retired,
        introduced_source_ids=introduced,
        retired_in_this_step=retired_now,
        gravity_present=child.spec.gravity is not None,
    )


def require_fresh_structural_ids(previous: SourceWorldAncestry, case) -> None:
    """Prequalify M22a incoming IDs against whole-graph, branch-local history."""
    if type(previous) is not SourceWorldAncestry:
        raise TypeError("missing native retained source ancestry certificate")
    # Caller already typed StructuralCase; do not treat ordinary JSON as a case.
    incoming = tuple(x.source_id for x in case.add_sources) + tuple(
        x.new.source_id for x in case.replace_sources)
    if any(i in previous.seen_source_ids for i in incoming):
        raise ValueError(
            "retired or existing source identity cannot be reused across "
            "structural/physical graph ancestry"
        )
