"""M34b/M35a: certified incremental and multi-stage physical-world forks from one genuine issued graph.

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


def _root_worlds(graph: UnifiedObservationRun):
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
class _SourceContext:
    """Authenticated per-call view; never stored on a public plan/result."""

    root: UnifiedObservationRun
    source: object
    root_digest: str
    record: dict
    worlds: dict[str, _World]
    depths: dict[str, int]
    stages: tuple[dict, ...] = ()


def _source_context(source):
    """Iterative bounded validation without replaying any ODE.

    Collect the untrusted object chain FIRST, reject cycles/depth/type errors,
    then validate root and each prior issued stage exactly once in this call.
    Work also depends on the saved samples and ancestry in those records.
    The per-call context is discarded; a later mutation cannot hit a cache.
    """
    chain = []
    seen = set()
    cursor = source
    while type(cursor) is GraphWorldForkRun:
        if id(cursor) in seen:
            raise ValueError("cycle in native exploration history")
        seen.add(id(cursor))
        if len(chain) >= 8:
            raise ValueError("exploration history exceeds eight fork stages")
        if type(cursor.plan) is not GraphWorldForkPlan:
            raise TypeError("fork history requires original native plans")
        chain.append(cursor)
        cursor = cursor.plan.source_graph
    if type(cursor) is not UnifiedObservationRun:
        raise TypeError("source history requires an issued native graph or fork")
    record = cursor.as_dict()  # existing independent root receipt validation
    if cursor.plan.retain_source_ancestry is not True:
        raise ValueError("fork requires source graph retained ancestry")
    depths = {"baseline": 0}
    for node in cursor.plan.nodes:
        depths[node.node_id] = depths[node.parent_id] + 1
    context = _SourceContext(cursor, cursor, _receipt(record), record,
                             _root_worlds(cursor), depths)
    for run in reversed(chain):
        if run.plan.source_graph is not context.source:
            raise ValueError("history lost its immediate native source run")
        record, worlds = _validate_run(run, context)
        depths = dict(context.depths)
        for node in run.plan.nodes:
            depths[node.node_id] = depths[node.parent_id] + 1
        stage = dict(fork_id=run.plan.fork_id, run_sha256=_receipt(record),
                     world_ids=tuple(n.node_id for n in run.plan.nodes))
        context = _SourceContext(cursor, run, context.root_digest, record,
                                 worlds, depths, context.stages + (stage,))
    return context


def _source_worlds(source):
    return _source_context(source).worlds


def _validate_plan(plan, context):
    if type(plan) is not GraphWorldForkPlan or plan.source_graph is not context.source:
        raise TypeError("fork requires an original native plan and authenticated source")
    if type(plan.fork_id) is not str or _ID.fullmatch(plan.fork_id) is None:
        raise ValueError("fork needs a bounded native ASCII ID")
    if len(context.stages) >= 8:
        raise ValueError("exploration history exceeds eight fork stages")
    if context.stages and plan.fork_id in (
            context.root.plan.plan_id, *(s["fork_id"] for s in context.stages)):
        raise ValueError("fork ID is already present in inherited history")
    max_worlds, max_depth = (32, 12) if context.stages else (12, 6)
    if (type(plan.nodes) is not tuple or not 1 <= len(plan.nodes) <= 5 or
            len(context.worlds) - 1 + len(plan.nodes) > max_worlds):
        raise ValueError("fork exceeds new or cumulative world-count budget")
    known, depths = set(context.worlds), dict(context.depths)
    model_ids = {w.model.scenario_id for w in context.worlds.values()
                 if w.model is not None}
    fresh = set()
    edits = 0
    for node in plan.nodes:
        if type(node) not in (StructuralWorldNode, PhysicalWorldNode):
            raise TypeError("fork nodes must be exact native physical/structural nodes")
        node.__post_init__()  # pure nested request validation, not model reconstruction
        if node.node_id in known:
            raise ValueError("fork cannot reuse an inherited or earlier world ID")
        if node.parent_id not in known:
            raise ValueError("fork parent must be an earlier actual source world")
        if depths[node.parent_id] + 1 > max_depth:
            raise ValueError("fork exceeds bounded physical source lineage depth")
        if f"{plan.fork_id}:{node.node_id}" in model_ids:
            raise ValueError("fork scenario identity collides with existing model")
        depths[node.node_id] = depths[node.parent_id] + 1
        known.add(node.node_id)
        fresh.add(node.node_id)
        if type(node) is PhysicalWorldNode:
            edits += len(node.edits)
    if edits > 12:
        raise ValueError("fork requires at most twelve actual physical edits")
    if type(plan.questions) is not tuple:
        raise TypeError("event questions require immutable native tuple")
    has_events = any(node.observation == "events" for node in plan.nodes)
    if has_events:
        validate_event_requests(plan.questions, plan.policy)
    elif plan.questions or plan.policy is not None:
        raise ValueError("fork has event policy without a new M24a event world")
    if type(plan.comparisons) is not tuple or len(plan.comparisons) > 8:
        raise ValueError("fork accepts at most eight immutable pair requests")
    pair_ids, unordered = set(), set()
    for pair in plan.comparisons:
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


@dataclass(frozen=True, slots=True)
class GraphWorldForkPlan:
    """A bounded native graph extension; prior physical worlds are NOT re-solved."""

    source_graph: UnifiedObservationRun | GraphWorldForkRun
    fork_id: str
    nodes: tuple[GraphNode, ...]
    questions: tuple[PlaneEvent | VelocityReversal, ...] = ()
    policy: EventPolicy | None = None
    comparisons: tuple[WorldComparison, ...] = ()

    def __post_init__(self):
        _validate_plan(self, _source_context(self.source_graph))

    def as_dict(self):
        context = _source_context(self.source_graph)
        _validate_plan(self, context)
        return json.loads(json.dumps(_plan_record(self, context), allow_nan=False))


def _plan_record(plan, context):
    source = context.record
    record = dict(
        schema="conductor_sim.graph_world_fork_plan.v1",
        fork_id=plan.fork_id,
        source_graph_plan_id=context.root.plan.plan_id,
        source_graph_schema=source["schema"],
        source_graph_sha256=_receipt(source),
        original_revision_id=context.root.plan.original.revision_id,
        source_world_ids=tuple(context.worlds),
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
    if context.stages:
        record.update(
            schema="conductor_sim.graph_world_fork_plan.v2",
            source_kind="prior-native-fork",
            original_graph_sha256=context.root_digest,
            previous_fork_id=context.stages[-1]["fork_id"],
            prior_stages=list(context.stages),
            stage_number=len(context.stages) + 1,
            cumulative_world_count=len(context.worlds) - 1 + len(plan.nodes),
            history_limits=dict(fork_stages=8, nonbaseline_worlds=32,
                                physical_depth=12, new_worlds_per_stage=5,
                                physical_edits_per_stage=12),
        )
    return record


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


def _record(plan, entries, context):
    graph = context.record
    worlds = dict(context.worlds)
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
            parent_origin=("this_fork" if parent_id not in context.worlds else
                           "prior_fork" if context.stages and parent_id not in
                           {"baseline", *(n.node_id for n in context.root.plan.nodes)}
                           else "source_graph"),
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
    record = dict(
        schema="conductor_sim.graph_world_fork.v1",
        plan=_plan_record(plan, context),
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
    if context.stages:
        record.update(
            schema="conductor_sim.graph_world_fork.v2",
            source_history_sha256=_receipt(graph),
            original_graph_sha256=context.root_digest,
            stage_number=len(context.stages) + 1,
            prior_stages=list(context.stages),
            cumulative_world_count=len(context.worlds) - 1 + len(entries),
        )
    return record


@dataclass(frozen=True, slots=True)
class GraphWorldForkRun:
    plan: GraphWorldForkPlan
    entries: tuple[GraphWorldForkEntry, ...]
    _issued_digest: str = field(default="", repr=False, compare=False)
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        _source_context(self)

    def as_dict(self):
        """Iteratively revalidate every ancestor; never retain a mutable cache."""
        return json.loads(json.dumps(_source_context(self).record, allow_nan=False))


def _validate_run(run, context):
    if type(run.plan) is not GraphWorldForkPlan or type(run.entries) is not tuple:
        raise TypeError("fork run requires original native plan and tuple of results")
    _validate_plan(run.plan, context)
    if len(run.entries) != len(run.plan.nodes):
        raise ValueError("fork run missing actual newly authored worlds")
    parent_worlds = dict(context.worlds)
    for index, (node, entry) in enumerate(zip(run.plan.nodes, run.entries), 1):
        if (type(entry) is not GraphWorldForkEntry or
                type(entry.index) is not int or entry.index != index or entry.node is not node or
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
                    run.plan.questions, run.plan.policy):
                raise ValueError("fork native event/trajectory status differs from real termination")
        parent_worlds[node.node_id] = _World(
            entry.scenario, entry.assembly, entry.native, node.observation,
            entry.status, entry.source_ancestry,
            run.plan.questions, run.plan.policy,
        )
    record = _record(run.plan, run.entries, context)
    if (run._issuer is not _ISSUER or type(run._issued_digest) is not str
            or not compare_digest(run._issued_digest, _receipt(record))):
        raise ValueError("fork evidence differs from original runner-issued receipt")
    return record, parent_worlds


def run_graph_world_fork(plan: GraphWorldForkPlan) -> GraphWorldForkRun:
    """Independently integrate ONLY new native worlds of a certified source graph."""
    if type(plan) is not GraphWorldForkPlan:
        raise TypeError("require native GraphWorldForkPlan, not imported JSON")
    context = _source_context(plan.source_graph)
    _validate_plan(plan, context)
    saved_digest = _receipt(context.record)
    worlds = dict(context.worlds)
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
    if _receipt(_source_context(plan.source_graph).record) != saved_digest:
        raise RuntimeError("original source history changed during new physical fork")
    issued = tuple(entries)
    return GraphWorldForkRun(
        plan, issued, _receipt(_record(plan, issued, context)), _ISSUER)
