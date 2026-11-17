"""M34b: certified incremental physical-world forks from one genuine issued graph.

Only NEW alternative source worlds are assembled, qualified and integrated.
Existing graph evidence is read/validated but never rebuilt or reintegrated.
Source and fork worlds are different immutable hypotheses, never a physical
transition, source motion, force-work integral or browser-side simulation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hmac import compare_digest
import json

from .coupled_physical_family import _inventory, _physical_changes
from .event_sweep import _root_pair
from .motion_1d import MotionIntegrationError
from .motion_events import (EventPolicy, EventRun, PlaneEvent, VelocityReversal,
                            validate_event_requests)
from .prescribed_motion import TestChargeMotionResult
from .scenarios import PrescribedScenario, revise_scenario
from .solver import UnsupportedProblem
from .source_world_ancestry import (
    SourceWorldAncestry, source_ancestry_root, advance_source_ancestry,
    require_fresh_structural_ids,
)
from .structural_scenarios import ScenarioReassembly, reassemble_scenario
from .trajectory_sweep import _shared_rows
from .unified_observation_graph import (
    _ID, _MODES, _NODE_ID, _observe, _receipt, _validate_native,
    GraphNode, PhysicalWorldNode, StructuralWorldNode, UnifiedObservationRun,
    WorldComparison,
)

_ISSUER = object()


@dataclass(frozen=True, slots=True)
class _World:
    """Actual source model and observation; not an editable evidence surrogate."""

    model: PrescribedScenario | None
    assembly: ScenarioReassembly | None
    native: TestChargeMotionResult | EventRun | None
    mode: str
    status: str
    ancestry: SourceWorldAncestry | None
    questions: tuple[PlaneEvent | VelocityReversal, ...]
    policy: EventPolicy | None


def _source_worlds(graph: UnifiedObservationRun):
    """Bind original and every existing node to runner-issued native identities."""
    source_questions, source_policy = graph.plan.questions, graph.plan.policy
    worlds = {
        "baseline": _World(
            graph.plan.original, None, graph.baseline_native,
            graph.plan.baseline_observation, graph.baseline_status,
            source_ancestry_root(graph.plan.original),
            source_questions, source_policy,
        )
    }
    for entry in graph.entries:
        worlds[entry.node.node_id] = _World(
            entry.scenario, entry.assembly, entry.native,
            entry.node.observation, entry.status, entry.source_ancestry,
            source_questions, source_policy,
        )
    return worlds


@dataclass(frozen=True, slots=True)
class GraphWorldForkPlan:
    """A bounded native graph extension; prior physical worlds are NOT re-solved."""

    source_graph: UnifiedObservationRun
    fork_id: str
    nodes: tuple[GraphNode, ...]
    questions: tuple[PlaneEvent | VelocityReversal, ...] = ()
    policy: EventPolicy | None = None
    comparisons: tuple[WorldComparison, ...] = ()

    def __post_init__(self):
        if type(self.source_graph) is not UnifiedObservationRun:
            raise TypeError("fork requires an authentic issued UnifiedObservationRun")
        self.source_graph.__post_init__()
        if self.source_graph.plan.retain_source_ancestry is not True:
            raise ValueError("fork requires source graph retained ancestry")
        if type(self.fork_id) is not str or _ID.fullmatch(self.fork_id) is None:
            raise ValueError("fork needs a bounded native ASCII ID")
        if (type(self.nodes) is not tuple or
                not 1 <= len(self.nodes) <= 5 or
                len(self.source_graph.plan.nodes) + len(self.nodes) > 12):
            raise ValueError("fork requires 1..5 nodes, at most 12 with source graph")
        known = {"baseline", *(node.node_id for node in self.source_graph.plan.nodes)}
        source_ids = {
            self.source_graph.plan.original.scenario_id,
            *(entry.scenario.scenario_id for entry in self.source_graph.entries
              if entry.scenario is not None),
        }
        depths = {"baseline": 0}
        for node in self.source_graph.plan.nodes:
            depths[node.node_id] = depths[node.parent_id] + 1
        physical_edits = 0
        fresh = set()
        for node in self.nodes:
            if type(node) not in (PhysicalWorldNode, StructuralWorldNode):
                raise TypeError("fork nodes must be exact native physical/structural nodes")
            node.__post_init__()
            if node.node_id in known or node.node_id in fresh:
                raise ValueError("fork cannot reuse original or earlier world IDs")
            if node.parent_id not in known:
                raise ValueError("fork parent must be an earlier actual source world")
            if depths[node.parent_id] + 1 > 6:
                raise ValueError("fork exceeds bounded physical source lineage depth")
            if f"{self.fork_id}:{node.node_id}" in source_ids:
                raise ValueError("fork scenario identity collides with existing model")
            depths[node.node_id] = depths[node.parent_id] + 1
            known.add(node.node_id)
            fresh.add(node.node_id)
            if type(node) is PhysicalWorldNode:
                physical_edits += len(node.edits)
        if physical_edits > 12:
            raise ValueError("fork requires at most twelve actual physical edits")
        if type(self.questions) is not tuple:
            raise TypeError("event questions require immutable native tuple")
        has_events = any(node.observation == "events" for node in self.nodes)
        if has_events:
            if (type(self.policy) is not EventPolicy or
                    not 1 <= len(self.questions) <= 16 or
                    any(type(q) not in (PlaneEvent, VelocityReversal)
                        for q in self.questions)):
                raise TypeError("new event fork worlds require native M24a questions/policy")
            if len({q.event_id for q in self.questions}) != len(self.questions):
                raise ValueError("duplicate event question IDs")
            validate_event_requests(self.questions, self.policy)
        elif self.questions or self.policy is not None:
            raise ValueError("fork has event policy without a new M24a event world")
        if type(self.comparisons) is not tuple or len(self.comparisons) > 8:
            raise ValueError("fork accepts at most eight immutable pair requests")
        pair_ids, unordered = set(), set()
        for pair in self.comparisons:
            if type(pair) is not WorldComparison:
                raise TypeError("fork pair must be genuine native WorldComparison")
            pair.__post_init__()
            if pair.left_world_id not in known or pair.right_world_id not in known:
                raise ValueError("fork comparison references unknown physical world")
            if not ({pair.left_world_id, pair.right_world_id} & fresh):
                raise ValueError("fork comparison must include a newly authored world")
            key = frozenset((pair.left_world_id, pair.right_world_id))
            if pair.comparison_id in pair_ids or key in unordered:
                raise ValueError("duplicate pair identity or unordered fork pair")
            pair_ids.add(pair.comparison_id)
            unordered.add(key)

    def as_dict(self):
        self.__post_init__()
        return json.loads(json.dumps(_plan_record(self, self.source_graph.as_dict()),
                                    allow_nan=False))


def _plan_record(plan, source):
    return dict(
        schema="conductor_sim.graph_world_fork_plan.v1",
        fork_id=plan.fork_id,
        source_graph_plan_id=plan.source_graph.plan.plan_id,
        source_graph_schema=source["schema"],
        source_graph_sha256=_receipt(source),
        original_revision_id=plan.source_graph.plan.original.revision_id,
        source_world_ids=("baseline",) + tuple(
            node.node_id for node in plan.source_graph.plan.nodes),
        nodes=[node.as_dict() for node in plan.nodes],
        questions=[dict(kind=type(question).__name__, **asdict(question))
                   for question in plan.questions],
        numerical_event_policy=(None if plan.policy is None
                                else asdict(plan.policy)),
        comparisons=[comparison.as_dict() for comparison in plan.comparisons],
        qualification=(
            "1..5 genuinely new whole-source physical hypotheses; prior source "
            "graph models and observations are reused exactly, never re-integrated"),
    )


@dataclass(frozen=True, slots=True)
class GraphWorldForkEntry:
    index: int
    node: GraphNode
    parent_scenario: PrescribedScenario | None
    scenario: PrescribedScenario | None
    assembly: ScenarioReassembly | None
    native: TestChargeMotionResult | EventRun | None
    status: str
    reason: str
    source_ancestry: SourceWorldAncestry | None


def _all_worlds(plan, entries):
    worlds = _source_worlds(plan.source_graph)
    for row in entries:
        worlds[row.node.node_id] = _World(
            row.scenario, row.assembly, row.native, row.node.observation,
            row.status, row.source_ancestry, plan.questions, plan.policy,
        )
    return worlds


def _study_signature(world):
    """Exact authored M24a question/policy meaning, not an unrelated event ID."""
    if world.mode != "events" or type(world.policy) is not EventPolicy:
        return None
    return _receipt(dict(
        questions=[dict(kind=type(q).__name__, **asdict(q)) for q in world.questions],
        policy=asdict(world.policy),
    ))


def _comparison_record(name, left_id, right_id, worlds, requested_questions):
    left, right = worlds[left_id], worlds[right_id]
    record = dict(
        comparison_id=name, left_world_id=left_id, right_world_id=right_id,
        numerical_orientation="right minus left",
        left_revision_id=None if left.model is None else left.model.revision_id,
        right_revision_id=None if right.model is None else right.model.revision_id,
        left_source_inventory=_inventory(left.model),
        right_source_inventory=_inventory(right.model),
        left_source_ancestry=(None if left.ancestry is None else left.ancestry.as_dict()),
        right_source_ancestry=(None if right.ancestry is None else right.ancestry.as_dict()),
        left_observation_mode=left.mode, right_observation_mode=right.mode,
        left_status=left.status, right_status=right.status,
        exact_time_trajectory_comparisons=[], event_time_comparisons=[],
        event_study_compatible=False,
    )
    why = ""
    if left.model is None or right.model is None:
        why = "at least one physical source world was not admitted"
    elif left.native is None or right.native is None:
        why = "at least one world lacks genuine native observation"
    elif left.mode != right.mode:
        why = "incompatible trajectory/event observation modes"
    else:
        ltraj = (left.native.trajectory if left.mode == "events" else left.native)
        rtraj = (right.native.trajectory if right.mode == "events" else right.native)
        record["exact_time_trajectory_comparisons"] = _shared_rows(ltraj, rtraj)
        if left.mode == "events":
            if _study_signature(left) == _study_signature(right):
                record["event_study_compatible"] = True
                record["event_time_comparisons"] = [
                    _root_pair(left.native, right.native, q.event_id)
                    for q in requested_questions]
            else:
                record["event_time_comparisons"] = [
                    dict(event_id=q.event_id, status="unpaired",
                         reason="different original/new M24a questions or numerical policies")
                    for q in requested_questions]
        if (not record["exact_time_trajectory_comparisons"] and
                not any(e["status"] == "one-resolved-pair"
                        for e in record["event_time_comparisons"])):
            why = "no shared actual saved SI time or compatible qualified event root"
    if why:
        record["status"], record["reason"] = "unpaired", why
        if not record["event_time_comparisons"] and requested_questions:
            record["event_time_comparisons"] = [
                dict(event_id=q.event_id, status="unpaired", reason=why)
                for q in requested_questions]
    else:
        record["status"] = "compared"
        record["reason"] = (
            "Actual separate native worlds; saved-time rows and each signed "
            "event pairing independently qualified; no physical source transition")
    return record


def _record(plan, entries):
    graph = plan.source_graph.as_dict()  # independently revalidate issued evidence
    worlds = _source_worlds(plan.source_graph)
    rows = []
    for entry in entries:
        parent_id, current_id = entry.node.parent_id, entry.node.node_id
        parent = worlds[parent_id]
        new = _World(entry.scenario, entry.assembly, entry.native,
                     entry.node.observation, entry.status, entry.source_ancestry,
                     plan.questions, plan.policy)
        compared = dict(worlds)
        compared[current_id] = new
        rows.append(dict(
            index=entry.index, node=entry.node.as_dict(),
            parent_id=parent_id,
            parent_origin=("source_graph" if parent_id == "baseline" or
                           parent_id in (n.node_id for n in plan.source_graph.plan.nodes)
                           else "this_fork"),
            parent_revision_id=(None if parent.model is None else parent.model.revision_id),
            parent_source_inventory=_inventory(parent.model),
            parent_source_ancestry=(None if parent.ancestry is None
                                    else parent.ancestry.as_dict()),
            status=entry.status, reason=entry.reason,
            admitted_scenario=(None if entry.scenario is None
                               else entry.scenario.as_dict()),
            source_inventory=_inventory(entry.scenario),
            source_ancestry=(None if entry.source_ancestry is None
                             else entry.source_ancestry.as_dict()),
            structural_assembly=(None if entry.assembly is None
                                 else entry.assembly.as_dict()),
            native_observation=(None if entry.native is None
                                else entry.native.as_dict()),
            native_event_outcomes=(None if type(entry.native) is not EventRun
                                   else entry.native.as_dict()["outcomes"]),
            parent_comparison=_comparison_record(
                "parent_" + str(entry.index), parent_id, current_id,
                compared, plan.questions),
        ))
        worlds[current_id] = new
    return dict(
        schema="conductor_sim.graph_world_fork.v1",
        plan=_plan_record(plan, graph),
        provenance=(
            "Explicit new teacher hypothetical physical alternatives derived "
            "from authentic earlier model revisions; NOT a continuous physical "
            "trajectory between distinct source worlds"),
        source_graph_sha256=_receipt(graph),
        source_graph_preserved=True,
        nodes=rows,
        cross_world_comparisons=[
            _comparison_record(pair.comparison_id, pair.left_world_id,
                               pair.right_world_id, worlds, plan.questions)
            for pair in plan.comparisons],
        time_alignment="literal shared native saved SI timestamps only",
        event_pairing=(
            "same genuine typed M24a questions and policy; one uniquely "
            "resolved same-direction root in each duration-complete run"),
        time_interpolation=False, postguard_extrapolation=False,
        qualification=(
            "New source worlds independently validated and integrated, old "
            "source graph and its existing native observations never re-solved; "
            "numerical root brackets are not exact ODE error bounds"),
    )


@dataclass(frozen=True, slots=True)
class GraphWorldForkRun:
    plan: GraphWorldForkPlan
    entries: tuple[GraphWorldForkEntry, ...]
    _issued_digest: str = field(default="", repr=False, compare=False)
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if type(self.plan) is not GraphWorldForkPlan or type(self.entries) is not tuple:
            raise TypeError("fork run requires original native plan and tuple of results")
        self.plan.__post_init__()
        if len(self.entries) != len(self.plan.nodes):
            raise ValueError("fork run missing actual newly authored worlds")
        parent_worlds = _source_worlds(self.plan.source_graph)
        for index, (node, entry) in enumerate(zip(self.plan.nodes, self.entries), 1):
            if (type(entry) is not GraphWorldForkEntry or
                    entry.index != index or entry.node is not node or
                    type(entry.status) is not str or type(entry.reason) is not str):
                raise ValueError("fork result changed genuine ordered node/status identity")
            parent = parent_worlds[node.parent_id]
            if entry.parent_scenario is not parent.model:
                raise ValueError("fork lost its actual immediate native model parent")
            if parent.model is None:
                if (entry.scenario is not None or entry.assembly is not None or
                        entry.native is not None or entry.source_ancestry is not None or
                        entry.status != "skipped-parent-model"):
                    raise ValueError("refused fork source parent produced forged child")
            elif type(node) is StructuralWorldNode:
                if entry.assembly is None:
                    if (entry.scenario is not None or entry.native is not None or
                            entry.source_ancestry is not None or
                            entry.status != "rejected-structural-model"):
                        raise ValueError("refused structural fork falsely admitted model")
                else:
                    actual_parent = parent.assembly if parent.assembly is not None else parent.model
                    if (type(entry.assembly) is not ScenarioReassembly or
                            entry.assembly.parent is not actual_parent or
                            entry.scenario is not entry.assembly.scenario or
                            entry.assembly.as_dict()["parent_revision_id"] !=
                            parent.model.revision_id):
                        raise ValueError("fork structural world lacks real immediate M22a parent")
            else:
                if entry.assembly is not None:
                    raise ValueError("physical fork cannot fake a M22a source assembly")
                if entry.scenario is None:
                    if (entry.native is not None or entry.source_ancestry is not None or
                            entry.status not in ("rejected-physical-edit",
                                                 "rejected-physical-model")):
                        raise ValueError("rejected physical fork has fictitious evidence")
                elif (type(entry.scenario) is not PrescribedScenario or
                      entry.scenario.parent_revision_id != parent.model.revision_id or
                      entry.scenario.source_annotations != parent.model.source_annotations):
                    raise ValueError("physical fork has wrong true source parent or inventory")
            if entry.scenario is not None:
                expected_ancestry = advance_source_ancestry(
                    parent.ancestry, world_id=node.node_id, parent_id=node.parent_id,
                    child=entry.scenario,
                    operation=("structural" if type(node) is StructuralWorldNode
                               else "physical"))
                if (type(entry.source_ancestry) is not SourceWorldAncestry or
                        entry.source_ancestry != expected_ancestry):
                    raise ValueError("fork branch-local source history/certificate forged")
                if entry.native is None:
                    allowed = (("integration-failed",) if node.observation == "trajectory"
                               else ("integration-failed", "rejected-event-study"))
                    if entry.status not in allowed:
                        raise ValueError("fork admitted model without native observation explanation")
                elif entry.status != _validate_native(
                        entry.native, entry.scenario, node.observation,
                        self.plan.questions, self.plan.policy):
                    raise ValueError("fork native event/trajectory status differs from real termination")
            parent_worlds[node.node_id] = _World(
                entry.scenario, entry.assembly, entry.native, node.observation,
                entry.status, entry.source_ancestry,
                self.plan.questions, self.plan.policy,
            )
        if (self._issuer is not _ISSUER or type(self._issued_digest) is not str
                or not compare_digest(self._issued_digest, _receipt(
                    _record(self.plan, self.entries)))):
            raise ValueError("fork evidence differs from original runner-issued receipt")

    def as_dict(self):
        """Revalidate both old and new native evidence before detached JSON export."""
        self.__post_init__()
        return json.loads(json.dumps(_record(self.plan, self.entries), allow_nan=False))


def run_graph_world_fork(plan: GraphWorldForkPlan) -> GraphWorldForkRun:
    """Independently integrate ONLY new native worlds of a certified source graph."""
    if type(plan) is not GraphWorldForkPlan:
        raise TypeError("require native GraphWorldForkPlan, not imported JSON")
    plan.__post_init__()
    saved_graph = plan.source_graph.as_dict()
    originals = _source_worlds(plan.source_graph)
    worlds = dict(originals)
    entries = []
    for index, node in enumerate(plan.nodes, 1):
        parent = worlds[node.parent_id]
        assembly, ancestry = None, None
        if parent.model is None:
            model, native, status, reason = (
                None, None, "skipped-parent-model",
                "Actual source parent model was not physically admitted")
        else:
            if type(node) is StructuralWorldNode:
                args = dict(
                    scenario_id=f"{plan.fork_id}:{node.node_id}",
                    note=node.case.note,
                    add_sources=node.case.add_sources,
                    remove_sources=node.case.remove_sources,
                    replace_sources=node.case.replace_sources,
                )
                if node.case.gravity_action == "add":
                    args["gravity"] = node.case.gravity
                elif node.case.gravity_action == "remove":
                    args["gravity"] = None
                actual_parent = (parent.assembly if parent.assembly is not None
                                 else parent.model)
                try:
                    require_fresh_structural_ids(parent.ancestry, node.case)
                    assembly = reassemble_scenario(actual_parent, **args)
                except (UnsupportedProblem, TypeError, ValueError) as exc:
                    model, native, status, reason = (
                        None, None, "rejected-structural-model",
                        f"{type(exc).__name__}: {exc}")
                else:
                    model = assembly.scenario
            else:
                try:
                    physical_patches = _physical_changes(parent.model, node.edits)
                except (UnsupportedProblem, TypeError, ValueError) as exc:
                    model, native, status, reason = (
                        None, None, "rejected-physical-edit",
                        f"{type(exc).__name__}: {exc}")
                else:
                    try:
                        model = revise_scenario(
                            parent.model, scenario_id=f"{plan.fork_id}:{node.node_id}",
                            note=node.note, **physical_patches)
                    except (UnsupportedProblem, TypeError, ValueError) as exc:
                        model, native, status, reason = (
                            None, None, "rejected-physical-model",
                            f"{type(exc).__name__}: {exc}")
            if model is not None:
                if type(node) is PhysicalWorldNode and model.parent_revision_id != parent.model.revision_id:
                    raise RuntimeError("fork M20b model revision parent does not match")
                if type(node) is StructuralWorldNode and assembly.as_dict()["parent_revision_id"] != parent.model.revision_id:
                    raise RuntimeError("fork M22a assembly parent revision does not match")
                ancestry = advance_source_ancestry(
                    parent.ancestry, world_id=node.node_id, parent_id=node.parent_id,
                    child=model,
                    operation=("structural" if type(node) is StructuralWorldNode
                               else "physical"))
                native, status, reason = _observe(
                    model, node.observation, plan.questions, plan.policy)
        entry = GraphWorldForkEntry(
            index, node, parent.model, model, assembly, native, status, reason,
            ancestry)
        entries.append(entry)
        worlds[node.node_id] = _World(
            model, assembly, native, node.observation, status, ancestry,
            plan.questions, plan.policy)
    if _receipt(plan.source_graph.as_dict()) != _receipt(saved_graph):
        raise RuntimeError("original source graph changed during new physical fork")
    issued = tuple(entries)
    return GraphWorldForkRun(
        plan, issued, _receipt(_record(plan, issued)), _ISSUER)
