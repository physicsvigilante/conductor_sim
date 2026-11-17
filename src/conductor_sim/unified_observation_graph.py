"""M32a: one finite graph of independently qualified physical-world observations.

Structural operations use original M22a reassembly and retained source ancestry.
One or several physical edits use original M20b atomic full-model qualification.
Each world is observed once with the unchanged M19/M27 motion integrator OR
the unchanged M24a event solver. There is no inferred source motion or ODE
interpolation, and ordinary physical histories never substitute for event runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from hmac import compare_digest
import json
import re

from .conservative_forces import UniformGravity
from .coupled_physical_family import CoupledPropertyEdit, _physical_changes, _inventory
from .event_sweep import _root_pair
from .motion_1d import MotionIntegrationError
from .motion_events import (
    EventPolicy, EventRun, EventStudy, PlaneEvent, VelocityReversal,
    investigate_motion, validate_event_run, validate_event_requests,
)
from .prescribed_motion import TestChargeMotionResult, integrate_test_charge
from .scenarios import PrescribedScenario, revise_scenario, _text
from .source_world_ancestry import (
    SourceWorldAncestry, source_ancestry_root, advance_source_ancestry,
    require_fresh_structural_ids,
)
from .solver import UnsupportedProblem
from .structural_family import StructuralCase
from .structural_scenarios import ScenarioReassembly, reassemble_scenario
from .trajectory_sweep import _shared_rows, _target


_ID = re.compile(r"[A-Za-z][A-Za-z0-9:_-]{0,35}\Z")
_NODE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,19}\Z")
_MODES = frozenset(("trajectory", "events"))
_EVENT_STATUSES = dict(
    duration="complete",
    **{"source-guard": "censored-source-guard",
       "terminal-event": "terminal-event", "event-unresolved": "event-unresolved"},
)
_ISSUER = object()


def _node_common(node_id, parent_id, observation):
    if type(node_id) is not str or _NODE_ID.fullmatch(node_id) is None or node_id == "baseline":
        raise ValueError("graph node needs a unique bounded ID other than baseline")
    if type(parent_id) is not str or (parent_id != "baseline" and
            _NODE_ID.fullmatch(parent_id) is None):
        raise ValueError("graph parent must be baseline or a previous real node ID")
    if type(observation) is not str or observation not in _MODES:
        raise ValueError("observation must be trajectory or events")


@dataclass(frozen=True, slots=True)
class StructuralWorldNode:
    """Apply one complete existing M22a inventory operation to an admitted world."""

    node_id: str
    parent_id: str
    case: StructuralCase
    observation: str = "trajectory"

    def __post_init__(self):
        _node_common(self.node_id, self.parent_id, self.observation)
        if type(self.case) is not StructuralCase or self.case.case_id != self.node_id:
            raise TypeError("structural node must bind same-ID native StructuralCase")
        self.case.__post_init__()
        for source in self.case.add_sources:
            source.__post_init__()
        for replacement in self.case.replace_sources:
            replacement.__post_init__()
            replacement.new.__post_init__()

    def as_dict(self):
        return dict(node_id=self.node_id, parent_id=self.parent_id,
                    operation="structural", case=self.case.as_dict(),
                    observation=self.observation)


@dataclass(frozen=True, slots=True)
class PhysicalWorldNode:
    """One atomic supported 1..4-property SI revision of an immediate parent."""

    node_id: str
    parent_id: str
    note: str
    edits: tuple[CoupledPropertyEdit, ...]
    observation: str = "trajectory"

    def __post_init__(self):
        _node_common(self.node_id, self.parent_id, self.observation)
        _text(self.note, "physical graph teacher hypothesis", 400)
        if type(self.edits) is not tuple or not 1 <= len(self.edits) <= 4:
            raise ValueError("graph physical revision needs 1..4 immutable edits")
        if any(type(e) is not CoupledPropertyEdit for e in self.edits):
            raise TypeError("graph physical edits must be native CoupledPropertyEdit")
        for edit in self.edits:
            edit.__post_init__()
        targets = [(e.component_id, e.property_name) for e in self.edits]
        if len(targets) != len(set(targets)):
            raise ValueError("graph physical node repeats a component-property target")

    def as_dict(self):
        return dict(node_id=self.node_id, parent_id=self.parent_id,
                    operation="physical", note=self.note,
                    edits=[e.as_dict() for e in self.edits],
                    observation=self.observation)


GraphNode = StructuralWorldNode | PhysicalWorldNode


@dataclass(frozen=True, slots=True)
class WorldComparison:
    """Explicit ordered alternatives: numerical differences mean right minus left."""

    comparison_id: str
    left_world_id: str
    right_world_id: str

    def __post_init__(self):
        if type(self.comparison_id) is not str or _NODE_ID.fullmatch(self.comparison_id) is None:
            raise ValueError("comparison needs a bounded native ASCII ID")
        for world_id in (self.left_world_id, self.right_world_id):
            if (type(world_id) is not str or
                    (world_id != "baseline" and _NODE_ID.fullmatch(world_id) is None)):
                raise ValueError("comparison needs exact bounded native world IDs")
        if self.left_world_id == self.right_world_id:
            raise ValueError("comparison requires two different physical worlds")

    def as_dict(self):
        return dict(comparison_id=self.comparison_id,
                    left_world_id=self.left_world_id,
                    right_world_id=self.right_world_id)


@dataclass(frozen=True, slots=True)
class UnifiedObservationPlan:
    """An original model and 1..8 topologically ordered native worlds."""

    plan_id: str
    original: PrescribedScenario
    nodes: tuple[GraphNode, ...]
    baseline_observation: str = "trajectory"
    questions: tuple[PlaneEvent | VelocityReversal, ...] = ()
    policy: EventPolicy | None = None
    retain_source_ancestry: bool = False
    comparisons: tuple[WorldComparison, ...] = ()

    def __post_init__(self):
        if type(self.plan_id) is not str or _ID.fullmatch(self.plan_id) is None:
            raise ValueError("graph plan ID must be stable ASCII and at most 36 characters")
        if type(self.original) is not PrescribedScenario:
            raise TypeError("graph must be rooted at one native PrescribedScenario")
        if type(self.retain_source_ancestry) is not bool:
            raise TypeError("retain_source_ancestry requires an exact native Boolean")
        if type(self.baseline_observation) is not str or self.baseline_observation not in _MODES:
            raise ValueError("baseline observation must be trajectory or events")
        if type(self.nodes) is not tuple or not 1 <= len(self.nodes) <= 8:
            raise ValueError("graph requires 1..8 immutable physical worlds")
        seen = {"baseline": (None, 0)}
        n_edits = 0
        for node in self.nodes:
            if type(node) not in (StructuralWorldNode, PhysicalWorldNode):
                raise TypeError("graph requires exact native structural or physical nodes")
            node.__post_init__()
            if node.node_id in seen:
                raise ValueError("graph contains duplicate world IDs")
            if node.parent_id not in seen:
                raise ValueError("graph parent must be previous world: no forward or cyclic edges")
            parent_type, depth = seen[node.parent_id]
            if depth + 1 > 4:
                raise ValueError("graph physical provenance depth exceeds four revisions")
            if (type(node) is StructuralWorldNode and node.parent_id != "baseline" and
                    parent_type is not StructuralWorldNode and
                    not self.retain_source_ancestry):
                raise ValueError("structural-after-physical is unsupported: retain M22a source ancestry")
            if f"{self.plan_id}:{node.node_id}" == self.original.scenario_id:
                raise ValueError("graph world must not reuse original source-model identity")
            if type(node) is PhysicalWorldNode:
                n_edits += len(node.edits)
            seen[node.node_id] = (type(node), depth + 1)
        if type(self.comparisons) is not tuple or len(self.comparisons) > 8:
            raise ValueError("graph comparisons require at most eight immutable requests")
        if self.comparisons and not self.retain_source_ancestry:
            raise ValueError("cross-world comparisons require retained source ancestry")
        comparison_ids = set()
        pairs = set()
        for comparison in self.comparisons:
            if type(comparison) is not WorldComparison:
                raise TypeError("comparison must be an exact native WorldComparison")
            comparison.__post_init__()
            if (comparison.left_world_id not in seen or
                    comparison.right_world_id not in seen):
                raise ValueError("comparison references an unknown graph world")
            pair = frozenset((comparison.left_world_id, comparison.right_world_id))
            if comparison.comparison_id in comparison_ids or pair in pairs:
                raise ValueError("duplicate comparison identity or unordered world pair")
            comparison_ids.add(comparison.comparison_id)
            pairs.add(pair)
        if n_edits > 20:
            raise ValueError("graph has more than 20 requested physical edits")

        has_events = (self.baseline_observation == "events" or
                      any(node.observation == "events" for node in self.nodes))
        if type(self.questions) is not tuple:
            raise TypeError("event questions must be one immutable tuple")
        if has_events:
            if (type(self.policy) is not EventPolicy or
                    not 1 <= len(self.questions) <= 16 or
                    any(type(q) not in (PlaneEvent, VelocityReversal)
                        for q in self.questions)):
                raise TypeError("event worlds require 1..16 existing M24a questions and policy")
            if len({q.event_id for q in self.questions}) != len(self.questions):
                raise ValueError("duplicate event identity")
            validate_event_requests(self.questions, self.policy)
        elif self.questions or self.policy is not None:
            raise ValueError("event questions/policy without any event observation")

    def as_dict(self):
        from dataclasses import asdict
        record = dict(
            schema="conductor_sim.unified_observation_plan.v1",
            plan_id=self.plan_id, original=self.original.as_dict(),
            original_revision_id=self.original.revision_id,
            baseline_observation=self.baseline_observation,
            nodes=[node.as_dict() for node in self.nodes],
            questions=[dict(kind=type(q).__name__, **asdict(q))
                       for q in self.questions],
            numerical_event_policy=(None if self.policy is None else asdict(self.policy)),
        )
        if self.retain_source_ancestry:
            record["schema"] = "conductor_sim.unified_observation_plan.v2"
            record["retain_source_ancestry"] = True
        if self.comparisons:
            record["schema"] = "conductor_sim.unified_observation_plan.v3"
            record["comparisons"] = [c.as_dict() for c in self.comparisons]
        return record


@dataclass(frozen=True, slots=True)
class UnifiedObservationEntry:
    index: int
    node: GraphNode
    parent_scenario: PrescribedScenario | None
    scenario: PrescribedScenario | None
    assembly: ScenarioReassembly | None
    native: TestChargeMotionResult | EventRun | None
    status: str
    reason: str
    source_ancestry: SourceWorldAncestry | None = None

    @property
    def trajectory(self):
        if type(self.native) is EventRun:
            return self.native.trajectory
        return self.native


def _observe(scenario, mode, questions, policy):
    if mode == "trajectory":
        try:
            native = integrate_test_charge(scenario.spec)
        except (UnsupportedProblem, MotionIntegrationError) as exc:
            return None, "integration-failed", f"{type(exc).__name__}: {exc}"
    else:
        try:
            study = EventStudy(scenario, questions, policy)
        except (UnsupportedProblem, ValueError, TypeError) as exc:
            return None, "rejected-event-study", f"{type(exc).__name__}: {exc}"
        try:
            native = investigate_motion(study)
        except (UnsupportedProblem, MotionIntegrationError) as exc:
            return None, "integration-failed", f"{type(exc).__name__}: {exc}"
    return native, _native_status(native, mode), native.trajectory.termination if mode == "events" else native.termination


def _native_status(native, mode):
    if mode == "trajectory":
        if type(native) is not TestChargeMotionResult or native.termination not in (
                "duration", "source-guard"):
            raise ValueError("unknown genuine ordinary physical trajectory termination")
        return "complete" if native.termination == "duration" else "censored-source-guard"
    if type(native) is not EventRun:
        raise TypeError("event evidence must be native EventRun")
    if native.trajectory.termination not in _EVENT_STATUSES:
        raise ValueError("unknown genuine native event termination")
    return _EVENT_STATUSES[native.trajectory.termination]


def _validate_native(native, scenario, mode, questions, policy):
    if mode == "trajectory":
        if (type(native) is not TestChargeMotionResult or
                native.spec is not scenario.spec):
            raise ValueError("graph trajectory does not match native physical world")
    elif (type(native) is not EventRun or type(native.study) is not EventStudy or
          type(native.trajectory) is not TestChargeMotionResult or
          native.study.scenario is not scenario or
          native.trajectory.spec is not scenario.spec or
          native.study.questions is not questions or
          native.study.policy is not policy):
        raise ValueError("graph event does not match native scenario/questions/policy")
    if mode == "events":
        validate_event_run(native)
    return _native_status(native, mode)


def _native_dict(native):
    return None if native is None else native.as_dict()


def _time_and_event_pairs(parent_native, parent_mode, child_native, child_mode,
                          questions):
    if parent_native is None or child_native is None:
        reason = "missing native parent or child observation"
    elif parent_mode != child_mode:
        reason = "different native trajectory/event observation modes"
    else:
        reason = ""
    if reason:
        return [], [
            dict(event_id=q.event_id, status="unpaired", reason=reason)
            for q in questions
        ] if child_mode == "events" else []
    p = (parent_native.trajectory if parent_mode == "events" else parent_native)
    c = (child_native.trajectory if child_mode == "events" else child_native)
    rows = _shared_rows(p, c)
    comparisons = ([_root_pair(parent_native, child_native, q.event_id)
                    for q in questions] if child_mode == "events" else [])
    return rows, comparisons


def _cross_world_records(plan, baseline_native, baseline_status, entries):
    """Actual independent native worlds, never a fabricated graph edge or solve."""
    if not plan.comparisons:
        return []
    root = source_ancestry_root(plan.original)
    by_id = {"baseline": (plan.original, baseline_native,
                           plan.baseline_observation, baseline_status, root)}
    for entry in entries:
        by_id[entry.node.node_id] = (entry.scenario, entry.native,
            entry.node.observation, entry.status, entry.source_ancestry)
    output = []
    for request in plan.comparisons:
        lmodel, lnative, lmode, lstatus, lanc = by_id[request.left_world_id]
        rmodel, rnative, rmode, rstatus, ranc = by_id[request.right_world_id]
        record = dict(**request.as_dict(),
            left_revision_id=None if lmodel is None else lmodel.revision_id,
            right_revision_id=None if rmodel is None else rmodel.revision_id,
            left_source_inventory=_inventory(lmodel),
            right_source_inventory=_inventory(rmodel),
            left_source_ancestry=None if lanc is None else lanc.as_dict(),
            right_source_ancestry=None if ranc is None else ranc.as_dict(),
            left_observation_mode=lmode, right_observation_mode=rmode,
            left_status=lstatus, right_status=rstatus,
            numerical_orientation="right minus left",
            exact_time_trajectory_comparisons=[], event_time_comparisons=[])
        if lmodel is None or rmodel is None:
            reason = "at least one physical source model was not admitted"
        elif lnative is None or rnative is None:
            reason = "at least one admitted model has no native observation"
        elif lmode != rmode:
            reason = "different native trajectory/event observation modes"
        else:
            rows, pairings = _time_and_event_pairs(
                lnative, lmode, rnative, rmode, plan.questions)
            record["exact_time_trajectory_comparisons"] = rows
            record["event_time_comparisons"] = pairings
            reason = "" if rows else "no identical actually saved SI timestamps"
        if reason:
            record["status"] = "unpaired"
            record["reason"] = reason
            if lmode == rmode == "events" and not record["event_time_comparisons"]:
                record["event_time_comparisons"] = [
                    dict(event_id=q.event_id, status="unpaired", reason=reason)
                    for q in plan.questions]
        else:
            record["status"] = "compared"
            record["reason"] = "native shared-time observations; event pairs independently qualified"
        output.append(record)
    return output


def _record(plan, baseline_native, baseline_status, baseline_reason, entries):
    original = plan.original
    root_ancestry = source_ancestry_root(original) if plan.retain_source_ancestry else None
    observations = {"baseline": (
        original, baseline_native, plan.baseline_observation, root_ancestry)}
    output = []
    for entry in entries:
        parent_id = entry.node.parent_id
        parent_model, parent_native, parent_mode, parent_ancestry = observations[parent_id]
        current_mode = entry.node.observation
        exact, pairings = _time_and_event_pairs(
            parent_native, parent_mode, entry.native, current_mode, plan.questions)
        if entry.scenario is None:
            exact, pairings = [], [
                dict(event_id=q.event_id, status="unpaired",
                     reason="physical source model was not admitted")
                for q in plan.questions
            ] if current_mode == "events" else []
        record = dict(
            index=entry.index,
            node=entry.node.as_dict(),
            parent_id=parent_id,
            parent_revision_id=(None if entry.parent_scenario is None
                                else entry.parent_scenario.revision_id),
            accepted_parent_inventory=_inventory(entry.parent_scenario),
            status=entry.status, reason=entry.reason,
            admitted_scenario=(None if entry.scenario is None
                               else entry.scenario.as_dict()),
            revised_source_inventory=_inventory(entry.scenario),
            structural_assembly=(None if entry.assembly is None
                                 else entry.assembly.as_dict()),
            native_observation=_native_dict(entry.native),
            native_event_outcomes=(None if type(entry.native) is not EventRun
                                   else entry.native.as_dict()["outcomes"]),
            exact_time_parent_comparisons=exact,
            parent_event_time_comparisons=pairings,
        )
        if plan.retain_source_ancestry:
            record["parent_source_ancestry"] = (
                None if parent_ancestry is None else parent_ancestry.as_dict())
            record["source_ancestry"] = (
                None if entry.source_ancestry is None
                else entry.source_ancestry.as_dict())
        output.append(record)
        observations[entry.node.node_id] = (
            entry.scenario, entry.native, current_mode, entry.source_ancestry)
    result = dict(
        schema="conductor_sim.unified_observation_graph.v1",
        mode=("One genuine physical world per node, reusing original M22a "
              "reassembly, M20b atomic physical revisions, M19/M27 motion "
              "and M24a typed event integrators"),
        provenance=("Author-asserted original and separately authored teacher "
                    "hypothetical source worlds; none is motion of a held source"),
        plan=plan.as_dict(), plan_id=plan.plan_id,
        original_revision_id=original.revision_id,
        original_inventory=_inventory(original),
        original_observation_mode=plan.baseline_observation,
        original_observation_status=baseline_status,
        original_observation_reason=baseline_reason,
        original_native_observation=_native_dict(baseline_native),
        original_event_outcomes=(None if type(baseline_native) is not EventRun
                                 else baseline_native.as_dict()["outcomes"]),
        nodes=output,
        time_alignment="literal equality of original native saved SI timestamps only",
        event_pairing=("unique resolved same-direction roots in two duration-complete "
                       "M24a event runs of immediate parent and child only"),
        time_interpolation=False, postguard_extrapolation=False,
        qualification=("Numerical root brackets are NOT bounds on exact ODE errors; "
                       "no postguard path, physical source response, free-3D "
                       "or cross-inventory energy-as-work is inferred."),
    )
    if plan.retain_source_ancestry:
        result["schema"] = "conductor_sim.unified_observation_graph.v2"
        result["source_ancestry_policy"] = (
            "Graph-rooted, branch-local history of ALL ever-used source IDs "
            "across both structural and physical revisions; forbid retirement "
            "resurrection and retain exact immediate native source parents.")
        result["original_source_ancestry"] = root_ancestry.as_dict()
    if plan.comparisons:
        result["schema"] = "conductor_sim.unified_observation_graph.v3"
        result["cross_world_comparisons"] = _cross_world_records(
            plan, baseline_native, baseline_status, entries)
        result["cross_world_comparison_rule"] = (
            "Independent qualified worlds; right minus left; literal shared native "
            "SI timestamps only; same-direction unique duration-complete M24a roots "
            "only; no cross-inventory energy-as-work or simulated source motion")
    return result


def _receipt(record):
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class UnifiedObservationRun:
    plan: UnifiedObservationPlan
    baseline_native: TestChargeMotionResult | EventRun | None
    baseline_status: str
    baseline_reason: str
    entries: tuple[UnifiedObservationEntry, ...]
    _issued_digest: str = field(default="", repr=False, compare=False)
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if (type(self.plan) is not UnifiedObservationPlan or
                type(self.entries) is not tuple or
                len(self.entries) != len(self.plan.nodes) or
                type(self.baseline_status) is not str or
                type(self.baseline_reason) is not str):
            raise ValueError("graph result must be bound to its native original plan")
        self.plan.__post_init__()
        if self.baseline_native is None:
            if self.baseline_status not in ("rejected-event-study", "integration-failed"):
                raise ValueError("graph original lacks a recorded genuine observation stage")
        elif self.baseline_status != _validate_native(
                self.baseline_native, self.plan.original,
                self.plan.baseline_observation, self.plan.questions, self.plan.policy):
            raise ValueError("graph original observation status mismatches actual native termination")
        indexed = {"baseline": (
            self.plan.original, None,
            source_ancestry_root(self.plan.original)
            if self.plan.retain_source_ancestry else None)}
        for index, (node, entry) in enumerate(zip(self.plan.nodes, self.entries), 1):
            if (type(entry) is not UnifiedObservationEntry or entry.index != index or
                    entry.node is not node or type(entry.status) is not str or
                    type(entry.reason) is not str):
                raise ValueError("graph result altered its original ordered node identity")
            parent_model, parent_assembly, parent_ancestry = indexed[node.parent_id]
            if entry.parent_scenario is not parent_model:
                raise ValueError("graph result lost exact immediate source world")
            if parent_model is None:
                if (entry.scenario is not None or entry.assembly is not None or
                        entry.native is not None or entry.status != "skipped-parent-model"):
                    raise ValueError("refused physical parent has a fabricated descendant")
            elif type(node) is StructuralWorldNode:
                if entry.assembly is None:
                    if (entry.scenario is not None or entry.native is not None or
                            entry.status != "rejected-structural-model"):
                        raise ValueError("refused structural world has forged native evidence")
                else:
                    expected = (
                        parent_assembly if parent_assembly is not None
                        else parent_model)
                    if (type(entry.assembly) is not ScenarioReassembly or
                            entry.assembly.parent is not expected or
                            entry.scenario is not entry.assembly.scenario):
                        raise ValueError("structural node lost real M22a parent/assembly")
                    if (entry.assembly.as_dict()["parent_revision_id"] !=
                            parent_model.revision_id):
                        raise ValueError("structural node lost source revision ancestry")
            else:
                if entry.assembly is not None:
                    raise ValueError("physical-property node cannot forge M22a assembly")
                if entry.scenario is None:
                    if (entry.native is not None or
                            entry.status not in ("rejected-physical-edit",
                                                 "rejected-physical-model")):
                        raise ValueError("rejected physical node fabricated native evidence")
                elif (type(entry.scenario) is not PrescribedScenario or
                      entry.scenario.parent_revision_id != parent_model.revision_id or
                      entry.scenario.source_annotations != parent_model.source_annotations):
                    raise ValueError("physical node lost its actual M20b immediate parent")
            if entry.scenario is not None:
                if entry.native is None:
                    if entry.status not in ("rejected-event-study", "integration-failed"):
                        raise ValueError("admitted graph world has unexplained missing observation")
                elif entry.status != _validate_native(
                        entry.native, entry.scenario, node.observation,
                        self.plan.questions, self.plan.policy):
                    raise ValueError("native graph observation status disagrees with termination")
            if self.plan.retain_source_ancestry:
                expected_ancestry = (
                    None if entry.scenario is None else
                    advance_source_ancestry(
                        parent_ancestry, world_id=node.node_id,
                        parent_id=node.parent_id, child=entry.scenario,
                        operation=("structural" if type(node) is StructuralWorldNode
                                   else "physical")))
                if (entry.source_ancestry is not None and
                        type(entry.source_ancestry) is not SourceWorldAncestry):
                    raise TypeError("graph ancestry requires native source certificate")
                if (entry.source_ancestry is None) != (expected_ancestry is None) or (
                        expected_ancestry is not None and
                        entry.source_ancestry != expected_ancestry):
                    raise ValueError(
                        "mixed-order graph ancestry disagrees with real immediate source history")
            elif entry.source_ancestry is not None:
                raise ValueError("legacy v1 graph must not claim v2 ancestry evidence")
            indexed[node.node_id] = (
                entry.scenario, entry.assembly, entry.source_ancestry)
        if (self._issuer is not _ISSUER or
                type(self._issued_digest) is not str or
                not compare_digest(self._issued_digest,
                                   _receipt(_record(
                                       self.plan, self.baseline_native, self.baseline_status,
                                       self.baseline_reason, self.entries)))):
            raise ValueError("unified graph evidence differs from original runner-issued receipt")

    def as_dict(self):
        """Revalidate before a finite detached JSON export; never trusted import."""
        self.__post_init__()
        return json.loads(json.dumps(
            _record(self.plan, self.baseline_native, self.baseline_status,
                    self.baseline_reason, self.entries),
            allow_nan=False))


def run_unified_observation_graph(plan: UnifiedObservationPlan) -> UnifiedObservationRun:
    """Independently qualify and observe every world without source motion."""
    if type(plan) is not UnifiedObservationPlan:
        raise TypeError("require native UnifiedObservationPlan, not imported JSON")
    plan.__post_init__()
    before = plan.original.as_dict()
    sha = plan.original.revision_id
    initial, status, reason = _observe(
        plan.original, plan.baseline_observation, plan.questions, plan.policy)
    source = {"baseline": (
        plan.original, None,
        source_ancestry_root(plan.original)
        if plan.retain_source_ancestry else None)}
    rows = []
    for index, node in enumerate(plan.nodes, 1):
        parent, parent_assembly, parent_ancestry = source[node.parent_id]
        assembly = None
        ancestry = None
        if parent is None:
            world, native, state, why = (
                None, None, "skipped-parent-model",
                "Immediate physical parent has no accepted source model")
        else:
            if type(node) is StructuralWorldNode:
                kw = dict(
                    scenario_id=f"{plan.plan_id}:{node.node_id}", note=node.case.note,
                    add_sources=node.case.add_sources,
                    remove_sources=node.case.remove_sources,
                    replace_sources=node.case.replace_sources,
                )
                if node.case.gravity_action == "add":
                    kw["gravity"] = node.case.gravity
                elif node.case.gravity_action == "remove":
                    kw["gravity"] = None
                native_parent = (
                    parent_assembly if parent_assembly is not None
                    else parent)
                try:
                    if plan.retain_source_ancestry:
                        require_fresh_structural_ids(parent_ancestry, node.case)
                    assembly = reassemble_scenario(native_parent, **kw)
                except (UnsupportedProblem, ValueError, TypeError) as exc:
                    world, native, state, why = (
                        None, None, "rejected-structural-model",
                        f"{type(exc).__name__}: {exc}")
                else:
                    world = assembly.scenario
            else:
                try:
                    patches = _physical_changes(parent, node.edits)
                except (UnsupportedProblem, TypeError, ValueError) as exc:
                    world, native, state, why = (
                        None, None, "rejected-physical-edit",
                        f"{type(exc).__name__}: {exc}")
                else:
                    try:
                        world = revise_scenario(
                            parent, scenario_id=f"{plan.plan_id}:{node.node_id}",
                            note=node.note, **patches)
                    except (UnsupportedProblem, ValueError, TypeError) as exc:
                        world, native, state, why = (
                            None, None, "rejected-physical-model",
                            f"{type(exc).__name__}: {exc}")
            if world is not None:
                if (type(node) is PhysicalWorldNode and
                        world.parent_revision_id != parent.revision_id):
                    raise RuntimeError("graph physical world lost real M20b immediate parent")
                if (type(node) is StructuralWorldNode and
                        assembly.as_dict()["parent_revision_id"] != parent.revision_id):
                    raise RuntimeError("graph source structure lost real M22a immediate parent")
                if plan.retain_source_ancestry:
                    ancestry = advance_source_ancestry(
                        parent_ancestry, world_id=node.node_id,
                        parent_id=node.parent_id, child=world,
                        operation=("structural" if type(node) is StructuralWorldNode
                                   else "physical"))
                native, state, why = _observe(
                    world, node.observation, plan.questions, plan.policy)
        rows.append(UnifiedObservationEntry(
            index, node, parent, world, assembly, native, state, why, ancestry))
        source[node.node_id] = (world, assembly, ancestry)
    if plan.original.revision_id != sha or plan.original.as_dict() != before:
        raise RuntimeError("original physical world mutated during graph exploration")
    entries = tuple(rows)
    return UnifiedObservationRun(
        plan, initial, status, reason, entries,
        _receipt(_record(plan, initial, status, reason, entries)), _ISSUER)
