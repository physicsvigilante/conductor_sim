"""Explicit leading self-capacitance model for distant, wired spherical conductors.

Not a FieldSolution: no spatial field map or finite-wire discharge is supplied.
Existing conductor components and ConnectionPlan are reused without modification.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from math import fsum, hypot, isclose, isfinite, pi, ulp
import json

from .connections import ConnectionPlan, ConductorTransfer, ElectricalLink
from .model import ElectrostaticProblem, FixedNetCharge, SolidSphere, SphericalShell
from .solver import UnsupportedProblem

APPROXIMATION = 'self-capacitance-leading-order'
MAX_GEOMETRIC_ROW_SUM = 0.05  # Admission policy, NOT a 5% error guarantee.
LIMITATIONS = (
    'Leading isolated-sphere capacitances: mutual potential and surface polarization omitted.',
    'Finite wire capacitance, geometry, resistance, current, discharge time and radiation not solved.',
    'Surface E and sigma are uniform leading-order values, not resolved angular distributions.',
    'Separation criterion is an admission policy, not a bound on relative physical error.',
    'Near-neutral totals or strongly unequal charges can have large relative errors in omitted effects.',
    'Released energy is available electrostatic energy on settling, not a computed heat/current history.',
)


def _finite(value, name):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f'{name} must be finite and non-boolean')
    return float(value)


def _layout(plan, approximation):
    if approximation != APPROXIMATION:
        raise UnsupportedProblem('explicit self-capacitance-leading-order approximation required')
    if type(plan) is not ConnectionPlan:
        raise TypeError('expected immutable ConnectionPlan')
    p = plan.initial
    if type(p) is not ElectrostaticProblem or p.charges:
        raise UnsupportedProblem('distant spheres require no point sources, including empty cavities')
    cs = p.conductors
    if not 2 <= len(cs) <= 8 or any(type(c) not in (SolidSphere, SphericalShell) for c in cs):
        raise UnsupportedProblem('require 2 to 8 separated solid spheres or empty spherical shells')
    _finite(p.permittivity_f_per_m, 'permittivity')
    for c in cs:
        if type(c.constraint) is not FixedNetCharge:
            raise UnsupportedProblem('initial conductors must be isolated FixedNetCharge; no ground or voltage source')
        _finite(c.constraint.charge_c, 'net charge')
        _finite(c.outer_radius_m, 'outer radius')
        # Translation must not make the radii numerically unresolvable.
        if max(ulp(x) for x in c.center_m) > c.outer_radius_m*1e-8:
            raise UnsupportedProblem('sphere radius is unresolved at the authored world coordinates')
    index = {c.name:i for i,c in enumerate(cs)}
    occupied = set()
    groups = []
    for link in plan.links:
        if type(link) is not ElectricalLink or link.mode != 'floating':
            raise UnsupportedProblem('M21d supports floating links only, not grounding')
        if any(name not in index for name in link.members):
            raise UnsupportedProblem('link references an unknown conductor identity')
        if occupied.intersection(link.members):
            raise UnsupportedProblem('electrical groups must be disjoint; combine a connected group explicitly')
        occupied.update(link.members)
        groups.append(tuple(index[name] for name in link.members))
    distances = [[0.]*len(cs) for _ in cs]
    for i, a in enumerate(cs):
        for j in range(i+1, len(cs)):
            b = cs[j]
            d = hypot(*(x-y for x,y in zip(a.center_m, b.center_m)))
            if not isfinite(d) or d <= a.outer_radius_m+b.outer_radius_m:
                raise UnsupportedProblem('spheres must be disjoint and external, never nested or touching')
            distances[i][j] = distances[j][i] = d
    rows = tuple(fsum(c.outer_radius_m/d for j,d in enumerate(distances[i]) if j != i)
                 for i,c in enumerate(cs))
    if max(rows) > MAX_GEOMETRIC_ROW_SUM:
        raise UnsupportedProblem('far-separation admission failed: sum(R_i/d_ij) must be <= 0.05')
    return cs, tuple(groups), distances, rows


@dataclass(frozen=True, slots=True)
class DistantSphereObservation:
    name: str
    kind: str
    center_m: tuple[float, float, float]
    inner_radius_m: float
    outer_radius_m: float
    capacitance_f: float
    charge_c: float
    potential_v: float
    mean_outer_sigma_c_m2: float
    outer_field_normal_v_m: float
    inner_surface_charge_c: float
    interior_field_v_m: float


@dataclass(frozen=True, slots=True)
class DistantSphereState:
    conductors: tuple[DistantSphereObservation, ...]
    self_field_energy_j: float
    total_charge_c: float
    omitted_monopole_potential_estimate_v: tuple[float, ...]
    omitted_pair_energy_abs_scale_j: float

    def by_name(self, name):
        return next(row for row in self.conductors if row.name == name)


def _state(cs, charges, eps, distances):
    rows = []
    for c,q in zip(cs, charges):
        r = c.outer_radius_m
        cap = 4*pi*eps*r
        if not isfinite(cap) or cap <= 0:
            raise ValueError('self capacitance is numerically unrepresentable')
        v = q/cap
        sigma = q/(4*pi*r*r)
        e = v/r  # Outside normal component; inside metal/cavity E is zero.
        for value in (v, sigma, e):
            _finite(value, 'computed sphere observable')
        rows.append(DistantSphereObservation(c.name, type(c).__name__, c.center_m,
                    c.inner_radius_m, r, cap, q, v, sigma, e, 0., 0.))
    k = 1/(4*pi*eps)
    energy = fsum(.5*r.charge_c*r.potential_v for r in rows)
    omitted_v = tuple(k*fsum(charges[j]/d for j,d in enumerate(ds) if j != i)
                      for i,ds in enumerate(distances))
    omitted_u = fsum(abs(k*charges[i]*charges[j]/distances[i][j])
                      for i in range(len(cs)) for j in range(i+1, len(cs)))
    total = fsum(charges)
    for value in (energy, omitted_u, total, *omitted_v):
        _finite(value, 'computed energy/charge/omitted-effect diagnostic')
    return DistantSphereState(tuple(rows), energy, total, omitted_v, omitted_u)


@dataclass(frozen=True, slots=True)
class DistantSphereConnectionResult:
    plan: ConnectionPlan
    approximation: str
    resolved_problem: ElectrostaticProblem
    before: DistantSphereState
    after: DistantSphereState
    transfers: tuple[ConductorTransfer, ...]
    released_field_energy_j: float
    geometric_row_sums: tuple[float, ...]
    pair_distances_m: tuple[tuple[float, ...], ...]

    def as_dict(self):
        """Detached SI audit; not a trusted saved-model import or FieldSolution."""
        record = dict(schema='conductor_sim.distant_sphere_connections.v1',
            approximation=self.approximation,
            analysis='two static equilibria, not time evolution',
            gauge='potential zero at infinity in the leading self-capacitance model',
            conductor_ids=[c.name for c in self.plan.initial.conductors],
            permittivity_f_m=self.plan.initial.permittivity_f_per_m,
            links=[asdict(link) for link in self.plan.links],
            before=asdict(self.before), after=asdict(self.after),
            transfers=[asdict(t) for t in self.transfers],
            released_field_energy_j=self.released_field_energy_j,
            qualification=dict(geometric_row_sums=self.geometric_row_sums,
                maximum_allowed_row_sum=MAX_GEOMETRIC_ROW_SUM,
                pair_distances_m=self.pair_distances_m,
                is_relative_error_bound=False),
            limitations=LIMITATIONS)
        return json.loads(json.dumps(record, allow_nan=False))


class DistantSphereConnectionBackend:
    """Floating linked groups in an explicitly selected leading approximation.

    Does not route to NestedShellBackend/FEMM, expose spatial V/E, or pretend
    a global uniform-surface superposition is an exact conducting equilibrium.
    """
    def supports(self, plan, *, approximation):
        try:
            _layout(plan, approximation)
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def solve(self, plan: ConnectionPlan, *, approximation: str) -> DistantSphereConnectionResult:
        cs, groups, distances, rows = _layout(plan, approximation)
        try:
            eps = plan.initial.permittivity_f_per_m
            old = tuple(float(c.constraint.charge_c) for c in cs)
            new = list(old)
            for group in groups:
                total = fsum(old[i] for i in group)
                radius = fsum(cs[i].outer_radius_m for i in group)
                for i in group:
                    new[i] = total*(cs[i].outer_radius_m/radius)
            before = _state(cs, old, eps, distances)
            after = _state(cs, new, eps, distances)
            # Nonnegative weighted variance avoids U_before - U_after cancellation.
            released = fsum(.5*((old[i]-new[i])/before.conductors[i].capacitance_f)*(old[i]-new[i])
                            for group in groups for i in group)
            _finite(released, 'released field energy')
            for group in groups:
                charge_scale = fsum(abs(old[i]) for i in group)
                residual = fsum(new[i]-old[i] for i in group)
                if abs(residual) > 32*ulp(charge_scale):
                    raise ValueError('linked-group charge conservation failed')
                vs = [after.conductors[i].potential_v for i in group]
                if any(not isclose(v,vs[0],rel_tol=2e-14,abs_tol=32*ulp(max(abs(x) for x in vs))) for v in vs):
                    raise ValueError('linked group failed equal-potential validation')
            if not isclose(before.self_field_energy_j-after.self_field_energy_j,
                           released, rel_tol=2e-12,
                           abs_tol=64*ulp(max(before.self_field_energy_j,after.self_field_energy_j))):
                raise ValueError('energy accounting failed within floating-point resolution')
            resolved = replace(plan.initial, conductors=tuple(
                replace(c, constraint=FixedNetCharge(q)) for c,q in zip(cs,new)))
            return DistantSphereConnectionResult(plan, approximation, resolved, before, after,
                tuple(ConductorTransfer(c.name,a,b,b-a) for c,a,b in zip(cs,old,new)),
                released, rows, tuple(tuple(row) for row in distances))
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError('distant-sphere calculation exceeds numerical range') from exc
