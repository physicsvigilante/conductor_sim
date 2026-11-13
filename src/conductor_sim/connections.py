"""Composable *static* electrical connections between concentric shell conductors.

An ElectricalLink joins named shells. Floating groups conserve the sum of
initially prescribed shell charges; grounded groups acquire/remove charge from
a 0 V reservoir. The constraints are analytically resolved into ordinary
per-shell net charges or ground constraints, so existing field solvers and the
HTML viewer stay unchanged. No conductor merger geometry or FEMM force claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isclose, pi
from typing import Literal

from .model import ElectrostaticProblem, FixedNetCharge, Grounded, SolidSphere, SphericalShell
from .nested_shells import NestedShellBackend, NestedShellSolution, _solve_linear, _sorted_shells
from .mixed_radial import MixedRadialBackend, MixedRadialSolution
from .solver import UnsupportedProblem


@dataclass(frozen=True, slots=True)
class ElectricalLink:
    """One wire connecting 2+ named shells, optionally tied to ideal ground.

    A floating link conserves the **sum** of its members' initial net charges.
    Its members must initially have individually prescribed FixedNetCharge.
    A grounded link fixes the common potential to 0 V and permits exchange
    with the ground reservoir. Members may initially be isolated or grounded.
    """

    members: tuple[str, ...]
    mode: Literal['floating', 'grounded'] = 'floating'

    def __post_init__(self) -> None:
        if not isinstance(self.members, (tuple, list)):
            raise TypeError('link members must be an ordered sequence of conductor names')
        members = tuple(self.members)
        if (len(members) < 2 or any(not isinstance(name, str) or not name.strip()
                                     for name in members)
                or len(set(members)) != len(members)):
            raise ValueError('a link must name at least two distinct nonempty conductors')
        if self.mode not in ('floating', 'grounded'):
            raise ValueError("link mode must be 'floating' or 'grounded'")
        object.__setattr__(self, 'members', members)


@dataclass(frozen=True, slots=True)
class ConnectionPlan:
    """Immutable authored initial configuration and physical wire connections."""

    initial: ElectrostaticProblem
    links: tuple[ElectricalLink, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.initial, ElectrostaticProblem):
            raise TypeError('initial must be an ElectrostaticProblem')
        if not isinstance(self.links, (list, tuple)):
            raise TypeError('links must be a sequence of ElectricalLink objects')
        links = tuple(self.links)
        if not links or any(not isinstance(link, ElectricalLink) for link in links):
            raise ValueError('a connection plan needs one or more ElectricalLink objects')
        object.__setattr__(self, 'links', links)


@dataclass(frozen=True, slots=True)
class ConductorTransfer:
    name: str
    before_c: float
    after_c: float
    received_c: float


@dataclass(frozen=True, slots=True)
class ConnectionResult:
    """Two genuine static equilibria and a *region* field-energy difference.

    If there are fixed source charges within the innermost cavity, the cavity
    electric-field energy is unchanged and omitted from both regional values.
    The difference is still exact for fixed geometry and point sources.
    No current or discharge time is inferred from this equilibrium comparison.
    """

    plan: ConnectionPlan
    resolved_problem: ElectrostaticProblem
    before: NestedShellSolution | MixedRadialSolution
    after: NestedShellSolution | MixedRadialSolution
    transfers: tuple[ConductorTransfer, ...]
    radial_field_energy_before_j: float
    radial_field_energy_after_j: float
    released_field_energy_j: float

    def visualization(self, *, resolution: int = 31, line_steps: int = 140,
                      seeds=None, profile_samples: int = 55):
        """Reuse the accepted static-state sampler, not a wiring-specific viewer."""
        from .visualization import sample_sweep
        backend = (MixedRadialBackend() if any(isinstance(c, SolidSphere)
                       for c in self.plan.initial.conductors) else NestedShellBackend())
        bundle = sample_sweep([
            ('Before connecting shells', self.plan.initial),
            ('After connecting shells', self.resolved_problem),
        ], backend, resolution=resolution, line_steps=line_steps,
           seeds=seeds, profile_samples=profile_samples)
        modes = ', '.join(link.mode + ': ' + '/'.join(link.members)
                          for link in self.plan.links)
        bundle['frames'][0]['notes'].append('Wire OPEN: ' + modes + '. This is an initial static equilibrium.')
        bundle['frames'][1]['notes'].append('Wire CONNECTED: ' + modes + '. This is the final static equilibrium, not a current transient.')
        energy_scope = ('Fixed cavity-field energy is omitted equally from both states.'
                        if self.plan.initial.charges else
                        'Charge-free conductor model: this is the full field-energy difference.')
        bundle['frames'][1]['notes'].append(
            'Energy released from gaps/exterior: {:.6g} J. {}'.format(
                self.released_field_energy_j, energy_scope))
        bundle['connection'] = {
            'schema': 'conductor_sim.connection_equilibria.v1',
            'links': [{'members': list(link.members), 'mode': link.mode}
                      for link in self.plan.links],
            'transfers': [{'name': t.name, 'before_c': t.before_c,
                           'after_c': t.after_c, 'received_c': t.received_c}
                          for t in self.transfers],
            'radial_field_energy_before_j': self.radial_field_energy_before_j,
            'radial_field_energy_after_j': self.radial_field_energy_after_j,
            'released_field_energy_j': self.released_field_energy_j,
            'cavity_field_energy_omitted': bool(self.plan.initial.charges),
            'playback_meaning': 'Two static equilibria before and after a wire connection; not a transient current simulation.',
        }
        return bundle


def _layout(plan: ConnectionPlan):
    if not isinstance(plan, ConnectionPlan):
        raise UnsupportedProblem('connected-shell solver requires a ConnectionPlan')
    problem = plan.initial
    if any(not isinstance(c, (SphericalShell, SolidSphere)) for c in problem.conductors):
        raise UnsupportedProblem('connected radial wiring requires spherical conductors only')
    shells = _sorted_shells(problem)
    if any(isinstance(c, SolidSphere) for c in shells):
        if not MixedRadialBackend().supports(problem):
            raise UnsupportedProblem('mixed connections require one innermost solid sphere, '
                                     '1–7 concentric shells, and no point sources')
    elif not (2 <= len(shells) <= 8) or not NestedShellBackend().supports(problem):
        raise UnsupportedProblem('connected shells require 2–8 nonintersecting concentric '
                                 'SphericalShells with sources only in the innermost cavity')
    indices = {shell.name: i for i, shell in enumerate(shells)}
    seen: set[str] = set()
    linked = []
    for link in plan.links:
        if any(name not in indices for name in link.members):
            raise UnsupportedProblem('a link refers to a nonexistent shell')
        if seen.intersection(link.members):
            raise UnsupportedProblem('a conductor cannot belong to overlapping electrical links')
        seen.update(link.members)
        members = tuple(indices[name] for name in link.members)
        if (link.mode == 'floating' and
                any(not isinstance(shells[i].constraint, FixedNetCharge) for i in members)):
            raise UnsupportedProblem('a floating wire cannot conserve charge on an initially grounded conductor')
        linked.append((link, members))
    return shells, tuple(linked), set(indices[name] for name in seen)


def _coefficients(shells: tuple[SphericalShell, ...], charge_in_cavity: float):
    """Exact radial response V_i/k = b_i + sum_j A_ij Q_j."""
    n = len(shells)
    gaps = [1/shells[k].outer_radius_m - 1/shells[k + 1].inner_radius_m
            for k in range(n - 1)]
    inv_outer = 1/shells[-1].outer_radius_m
    A = [[inv_outer + fsum(gaps[k] for k in range(i, n - 1) if j <= k)
          for j in range(n)] for i in range(n)]
    b = [charge_in_cavity * (inv_outer + fsum(gaps[i:])) for i in range(n)]
    return A, b


def _field_energy(solution: NestedShellSolution | MixedRadialSolution) -> float:
    """Electric field energy in gaps + exterior, excluding fixed cavity field."""
    shells = solution.radial_shells
    flux = solution._fluxes()
    terms = [flux[k]**2 * (1/shells[k].outer_radius_m - 1/shells[k+1].inner_radius_m)
             for k in range(len(shells)-1)]
    terms.append(flux[-1]**2 / shells[-1].outer_radius_m)
    return fsum(terms)/(8*pi*solution.permittivity_f_per_m)


class ConnectedShellBackend:
    """Resolve wire constraints before using the standard nested-shell solver.

    Only the analytical concentric-shell domain is established. Compiling an
    equilibrium to individual FixedNetCharge/Grounded constraints does **not**
    validate a real FEMM solve of a physical connecting wire.
    """

    def supports(self, plan: ConnectionPlan) -> bool:
        try:
            _layout(plan)
            return True
        except (TypeError, ValueError, UnsupportedProblem):
            return False

    def solve(self, plan: ConnectionPlan) -> ConnectionResult:
        shells, links, grouped = _layout(plan)  # before any solver execution
        original = plan.initial
        solver = (MixedRadialBackend() if isinstance(shells[0], SolidSphere)
                  else NestedShellBackend())
        before = solver.solve(original)
        q_cavity = before.charge_inside_c
        A, b = _coefficients(shells, q_cavity)
        n = len(shells)
        matrix: list[list[float]] = []
        rhs: list[float] = []

        for link, members in links:
            reference = members[0]
            for i in members[1:]:
                matrix.append([A[i][j] - A[reference][j] for j in range(n)])
                rhs.append(b[reference] - b[i])
            if link.mode == 'floating':
                matrix.append([1. if j in members else 0. for j in range(n)])
                rhs.append(fsum(shells[i].constraint.charge_c for i in members))
            else:
                matrix.append(A[reference][:])
                rhs.append(-b[reference])
        for i, shell in enumerate(shells):
            if i in grouped:
                continue
            if isinstance(shell.constraint, Grounded):
                matrix.append(A[i][:])
                rhs.append(-b[i])
            else:
                matrix.append([1. if j == i else 0. for j in range(n)])
                rhs.append(shell.constraint.charge_c)
        if len(matrix) != n:
            raise UnsupportedProblem('connection constraints do not determine one net charge per shell')
        net = _solve_linear(matrix, rhs)
        replacements = {}
        for i, shell in enumerate(shells):
            # Keep grounded members grounded when lowering; the existing
            # analytical and numerical backends then preserve 0 V semantics.
            grounded_link = any(link.mode == 'grounded' and i in members
                                for link, members in links)
            replacements[shell.name] = Grounded() if grounded_link else FixedNetCharge(net[i]) if i in grouped else shell.constraint
        from dataclasses import replace
        resolved = ElectrostaticProblem(
            charges=original.charges,
            conductors=tuple(replace(shell, constraint=replacements[shell.name])
                             for shell in original.conductors),
            permittivity_f_per_m=original.permittivity_f_per_m,
        )
        after = solver.solve(resolved)
        for link, members in links:
            voltage = [after.conductor_by_name(shells[i].name).potential_v for i in members]
            if not all(isclose(v, voltage[0], rel_tol=1e-10, abs_tol=5e-8) for v in voltage):
                raise UnsupportedProblem('connected group has inconsistent solved potentials')
            if link.mode == 'grounded' and not isclose(voltage[0], 0., abs_tol=5e-8):
                raise UnsupportedProblem('grounded group has nonzero potential')
            if link.mode == 'floating':
                before_total = fsum(before.conductor_by_name(shells[i].name).net_charge_c for i in members)
                after_total = fsum(after.conductor_by_name(shells[i].name).net_charge_c for i in members)
                if not isclose(before_total, after_total, rel_tol=1e-10, abs_tol=1e-20):
                    raise UnsupportedProblem('floating group failed net-charge conservation')
        transfers = tuple(ConductorTransfer(shell.name,
                          before.conductor_by_name(shell.name).net_charge_c,
                          after.conductor_by_name(shell.name).net_charge_c,
                          after.conductor_by_name(shell.name).net_charge_c - before.conductor_by_name(shell.name).net_charge_c)
                          for shell in original.conductors)
        u0, u1 = _field_energy(before), _field_energy(after)
        released = u0-u1
        if released < -1e-11*max(u0, u1, 1e-20):
            raise UnsupportedProblem('connected equilibrium increased the radial electric-field energy')
        return ConnectionResult(plan, resolved, before, after, transfers, u0, u1,
                                max(0., released))
