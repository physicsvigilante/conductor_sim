"""M33b: re-ask genuine native event questions of qualified graph worlds.

An inquiry is a new *measurement* of selected already-qualified source models:
each admitted world receives a new genuine M24a EventStudy/investigate_motion.
No prior M19 trajectory is converted into a fabricated event history; no
source graph is rebuilt, no held source moves and no browser interpolates ODEs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from hmac import compare_digest
import json
import re

from .coupled_physical_family import _inventory
from .event_sweep import _root_pair
from .motion_events import EventPolicy, EventRun, PlaneEvent, VelocityReversal
from .scenarios import PrescribedScenario
from .source_world_ancestry import SourceWorldAncestry, source_ancestry_root
from .trajectory_sweep import _shared_rows
from .unified_observation_graph import (
    UnifiedObservationRun, WorldComparison, _observe, _validate_native,
)

_ID = re.compile(r"[A-Za-z][A-Za-z0-9:_-]{0,35}\Z")
_ISSUER = object()


def _receipt(value):
    """In-process record-seal, not a public cryptographic signature."""
    return sha256(json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")).hexdigest()


def _worlds(graph):
    """Real qualified graph scenarios, not reconstructed author assertions."""
    worlds = {"baseline": (
        graph.plan.original, source_ancestry_root(graph.plan.original),
        graph.baseline_status)}
    for row in graph.entries:
        worlds[row.node.node_id] = (
            row.scenario, row.source_ancestry, row.status)
    return worlds


def _plan_record(plan, source):
    return dict(
        schema="conductor_sim.graph_event_inquiry_plan.v1",
        inquiry_id=plan.inquiry_id,
        source_graph_plan_id=plan.graph.plan.plan_id,
        source_graph_schema=source["schema"],
        source_graph_sha256=_receipt(source),
        original_revision_id=plan.graph.plan.original.revision_id,
        world_ids=plan.world_ids,
        questions=[dict(kind=type(q).__name__, **asdict(q))
                   for q in plan.questions],
        numerical_event_policy=asdict(plan.policy),
        comparisons=[pair.as_dict() for pair in plan.comparisons],
        source_model_policy=(
            "Retain genuinely admitted native graph worlds and graph-root "
            "branch-local source ancestry; new event measurement, no reassembly"),
    )


@dataclass(frozen=True, slots=True)
class GraphEventInquiryPlan:
    """New question set for 1..8 actual qualified worlds of one issued graph."""

    graph: UnifiedObservationRun
    inquiry_id: str
    world_ids: tuple[str, ...]
    questions: tuple[PlaneEvent | VelocityReversal, ...]
    policy: EventPolicy
    comparisons: tuple[WorldComparison, ...] = ()

    def __post_init__(self):
        if type(self.graph) is not UnifiedObservationRun:
            raise TypeError("event inquiry requires an original runner-issued native graph")
        # Authenticate original graph *now*, before accepting an inquiry request.
        self.graph.__post_init__()
        if not self.graph.plan.retain_source_ancestry:
            raise ValueError("event inquiries require retained graph-root source ancestry")
        if type(self.inquiry_id) is not str or _ID.fullmatch(self.inquiry_id) is None:
            raise ValueError("event inquiry needs a bounded stable ASCII inquiry ID")
        if (type(self.world_ids) is not tuple or
                not 1 <= len(self.world_ids) <= 8 or
                any(type(world_id) is not str for world_id in self.world_ids) or
                len(set(self.world_ids)) != len(self.world_ids)):
            raise ValueError("event inquiry requires 1..8 distinct native world IDs")
        known = {"baseline", *(node.node_id for node in self.graph.plan.nodes)}
        if any(world_id not in known for world_id in self.world_ids):
            raise ValueError("event inquiry references a nonexistent physical world")
        if (type(self.questions) is not tuple or
                not 1 <= len(self.questions) <= 16 or
                any(type(question) not in (PlaneEvent, VelocityReversal)
                    for question in self.questions)):
            raise TypeError("event inquiry requires 1..16 genuine M24a typed questions")
        if len({question.event_id for question in self.questions}) != len(self.questions):
            raise ValueError("duplicate event question identities")
        if type(self.policy) is not EventPolicy:
            raise TypeError("event inquiry requires an exact native EventPolicy")
        if type(self.comparisons) is not tuple or len(self.comparisons) > 8:
            raise ValueError("event inquiry allows at most 8 native pair requests")
        pairs, pair_ids = set(), set()
        for pair in self.comparisons:
            if type(pair) is not WorldComparison:
                raise TypeError("event inquiry comparison requires WorldComparison")
            if pair.left_world_id not in self.world_ids or pair.right_world_id not in self.world_ids:
                raise ValueError("event inquiry comparison references an unselected world")
            edge = frozenset((pair.left_world_id, pair.right_world_id))
            if pair.comparison_id in pair_ids or edge in pairs:
                raise ValueError("duplicate event inquiry comparison identity or unordered pair")
            pairs.add(edge)
            pair_ids.add(pair.comparison_id)

    def as_dict(self):
        self.__post_init__()
        return json.loads(json.dumps(
            _plan_record(self, self.graph.as_dict()), allow_nan=False))


@dataclass(frozen=True, slots=True)
class GraphEventInquiryEntry:
    world_id: str
    scenario: PrescribedScenario | None
    source_ancestry: SourceWorldAncestry | None
    original_graph_observation_status: str
    native: EventRun | None
    status: str
    reason: str


def _comparison_record(request, selected, questions):
    left = selected[request.left_world_id]
    right = selected[request.right_world_id]
    out = dict(
        **request.as_dict(),
        numerical_orientation="right minus left",
        left_revision_id=(None if left.scenario is None else left.scenario.revision_id),
        right_revision_id=(None if right.scenario is None else right.scenario.revision_id),
        left_source_inventory=_inventory(left.scenario),
        right_source_inventory=_inventory(right.scenario),
        left_source_ancestry=(None if left.source_ancestry is None
                              else left.source_ancestry.as_dict()),
        right_source_ancestry=(None if right.source_ancestry is None
                               else right.source_ancestry.as_dict()),
        left_inquiry_status=left.status,
        right_inquiry_status=right.status,
        exact_time_trajectory_comparisons=[],
        event_time_comparisons=[],
    )
    if left.scenario is None or right.scenario is None:
        why = "at least one physical source world was refused"
    elif left.native is None or right.native is None:
        why = "at least one inquiry lacks genuine M24a event evidence"
    else:
        out["exact_time_trajectory_comparisons"] = _shared_rows(
            left.native.trajectory, right.native.trajectory)
        out["event_time_comparisons"] = [
            _root_pair(left.native, right.native, question.event_id)
            for question in questions]
        # Shared native positions and complete-history event root shifts are
        # independent observation types; one need not imply the other.
        if (out["exact_time_trajectory_comparisons"] or
                any(item["status"] == "one-resolved-pair"
                    for item in out["event_time_comparisons"])):
            why = ""
        else:
            why = "no shared saved SI times or qualified complete-history event pair"
    if why:
        out["status"] = "unpaired"
        out["reason"] = why
        if not out["event_time_comparisons"]:
            out["event_time_comparisons"] = [
                dict(event_id=question.event_id, status="unpaired", reason=why)
                for question in questions]
    else:
        out["status"] = "compared"
        out["reason"] = (
            "Independent M24a studies; shared SI samples and event pairings "
            "individually qualified; source worlds are alternatives")
    return out


def _record(plan, entries):
    source = plan.graph.as_dict()  # validates original runner evidence again
    ordered = {entry.world_id: entry for entry in entries}
    rows = []
    for entry in entries:
        rows.append(dict(
            world_id=entry.world_id,
            original_graph_observation_status=entry.original_graph_observation_status,
            model_revision_id=(None if entry.scenario is None
                               else entry.scenario.revision_id),
            source_inventory=_inventory(entry.scenario),
            source_ancestry=(None if entry.source_ancestry is None
                             else entry.source_ancestry.as_dict()),
            status=entry.status, reason=entry.reason,
            native_event_run=None if entry.native is None else entry.native.as_dict(),
            native_event_outcomes=None if entry.native is None
                                  else entry.native.as_dict()["outcomes"],
        ))
    return dict(
        schema="conductor_sim.graph_event_inquiry.v1",
        plan=_plan_record(plan, source),
        provenance=(
            "New teacher-authored measurement questions about existing genuine "
            "qualified alternative source worlds, not source-motion histories"),
        nodes=rows,
        cross_world_comparisons=[
            _comparison_record(pair, ordered, plan.questions)
            for pair in plan.comparisons],
        time_alignment="literally identical saved SI timestamps only",
        event_pairing=(
            "one unique resolved same-direction M24a root in EACH of two "
            "duration-complete new inquiry studies; numerical brackets not ODE errors"),
        time_interpolation=False,
        postguard_extrapolation=False,
        qualification=(
            "Each admitted selected model is re-integrated by native M24a for "
            "these NEW typed questions and numerical policy; never derive "
            "event roots from previous M19 saved trajectories. No physical "
            "source response or energy-as-work across alternative inventories."),
    )


@dataclass(frozen=True, slots=True)
class GraphEventInquiryRun:
    plan: GraphEventInquiryPlan
    entries: tuple[GraphEventInquiryEntry, ...]
    _issued_digest: str = field(default="", repr=False, compare=False)
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if type(self.plan) is not GraphEventInquiryPlan or type(self.entries) is not tuple:
            raise TypeError("inquiry result requires an original native inquiry plan/entries")
        self.plan.__post_init__()  # includes source graph receipt revalidation
        if len(self.entries) != len(self.plan.world_ids):
            raise ValueError("inquiry result is missing selected source worlds")
        genuine = _worlds(self.plan.graph)
        for world_id, entry in zip(self.plan.world_ids, self.entries):
            if (type(entry) is not GraphEventInquiryEntry or
                    type(entry.world_id) is not str or entry.world_id != world_id or
                    type(entry.status) is not str or type(entry.reason) is not str or
                    type(entry.original_graph_observation_status) is not str):
                raise ValueError("inquiry result has altered ordered native world IDs/status")
            model, ancestry, previous_status = genuine[world_id]
            if (entry.scenario is not model or
                    type(entry.source_ancestry) is not type(ancestry) or
                    entry.source_ancestry != ancestry or
                    entry.original_graph_observation_status != previous_status):
                raise ValueError("inquiry result lost genuine source-world model/ancestry")
            if model is None:
                if entry.native is not None or entry.status != "skipped-parent-model":
                    raise ValueError("refused physical model fabricated native event inquiry")
            elif entry.native is None:
                if entry.status not in ("rejected-event-study", "integration-failed"):
                    raise ValueError("admitted inquiry model has unexplained missing M24a run")
            elif entry.status != _validate_native(
                    entry.native, model, "events", self.plan.questions,
                    self.plan.policy):
                raise ValueError("new inquiry status differs from actual M24a event termination")
        if (self._issuer is not _ISSUER or
                type(self._issued_digest) is not str or
                not compare_digest(self._issued_digest, _receipt(
                    _record(self.plan, self.entries)))):
            raise ValueError("event inquiry differs from runner-issued in-process receipt")

    def as_dict(self):
        """Revalidate native physical/event evidence before any detached export."""
        self.__post_init__()
        return json.loads(json.dumps(_record(self.plan, self.entries), allow_nan=False))


def run_graph_event_inquiry(plan: GraphEventInquiryPlan) -> GraphEventInquiryRun:
    """Never reauthor source worlds: perform ONE new M24a study per admitted world."""
    if type(plan) is not GraphEventInquiryPlan:
        raise TypeError("event inquiry runner requires native GraphEventInquiryPlan")
    plan.__post_init__()
    original_evidence = plan.graph.as_dict()
    source = _worlds(plan.graph)
    entries = []
    for world_id in plan.world_ids:
        model, ancestry, prior_status = source[world_id]
        if model is None:
            native, status, reason = (
                None, "skipped-parent-model",
                "No admitted physical source world; cannot ask a native detector question")
        else:
            native, status, reason = _observe(
                model, "events", plan.questions, plan.policy)
        entries.append(GraphEventInquiryEntry(
            world_id, model, ancestry, prior_status, native, status, reason))
    if _receipt(plan.graph.as_dict()) != _receipt(original_evidence):
        raise RuntimeError("original graph evidence mutated during new inquiry")
    observed = tuple(entries)
    return GraphEventInquiryRun(
        plan, observed, _receipt(_record(plan, observed)), _ISSUER)
