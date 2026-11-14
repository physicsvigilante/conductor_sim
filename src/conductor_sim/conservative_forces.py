"""Bounded composition of prescribed electric force and uniform gravity.

No ODE solver, source-response model, equivalent electric field or arbitrary
force callbacks live here. Each component owns its force and potential energy;
the shared motion controller receives their sum without losing the ledger.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum, hypot, isclose, isfinite, ulp

from .model import Vector3
from .observables import EnergyBreakdown
from .prescribed_fields import PrescribedField, _number, _vector, _dot, _sub, _mul

ELECTRIC_ENERGY = 'U=q*V; source self/mutual energies excluded; no induced-energy half factor'
COMPOSED_ENERGY = 'U=q*V-m*g.dot(r-r_ref); electric and gravitational terms separate; no source self/mutual energy'
CONTRIBUTION_KEYS = ('electric_force_n', 'gravitational_force_n',
                     'electric_potential_energy_j', 'gravitational_potential_energy_j')


@dataclass(frozen=True, slots=True)
class UniformGravity:
    """Constant gravitational acceleration; reference_position_m fixes Ug=0.

    Acceleration is explicit, not implicitly Earth gravity. The reference is
    only an energy gauge, not an initial position or mechanical constraint.
    """
    acceleration_m_s2: Vector3
    reference_position_m: Vector3 = (0., 0., 0.)

    def __post_init__(self):
        for key in ('acceleration_m_s2', 'reference_position_m'):
            object.__setattr__(self, key, _vector(getattr(self, key), key))
        if not isfinite(hypot(*self.acceleration_m_s2)):
            raise ValueError('uniform-gravity magnitude must be numerically representable')

    def force_at(self, position_m, *, mass_kg):
        _vector(position_m, 'gravity observation position')
        _number(mass_kg, 'mass_kg', positive=True)
        force = _vector(_mul(self.acceleration_m_s2, mass_kg), 'gravitational force')
        if any(a != 0. and f == 0. for a, f in zip(self.acceleration_m_s2, force)):
            raise ValueError('gravitational force underflows the numerical representation')
        return force

    def potential_energy_at(self, position_m, *, mass_kg):
        p = _vector(position_m, 'gravity observation position')
        _number(mass_kg, 'mass_kg', positive=True)
        return _number(-mass_kg * _dot(self.acceleration_m_s2, _sub(p, self.reference_position_m)),
                       'gravitational potential energy')

    def as_dict(self):
        return dict(kind='UniformGravity', **asdict(self))


def restore_gravity(record):
    """Strict data boundary: never import a callback or infer missing gravity."""
    if (not isinstance(record, dict) or set(record) !=
            {'kind', 'acceleration_m_s2', 'reference_position_m'} or
            record['kind'] != 'UniformGravity'):
        raise ValueError('invalid uniform-gravity component record')
    return UniformGravity(record['acceleration_m_s2'], record['reference_position_m'])


@dataclass(frozen=True, slots=True)
class PrescribedElectricForce:
    field: PrescribedField

    def __post_init__(self):
        if type(self.field) is not PrescribedField:
            raise TypeError('electric force requires a PrescribedField')

    def force_at(self, position_m, *, charge_c):
        return _mul(self.field.electric_field_at(position_m), charge_c)

    def potential_energy_at(self, position_m, *, charge_c):
        return charge_c * self.field.potential_at(position_m)


@dataclass(frozen=True, slots=True)
class ForceObservation:
    electric_force_n: Vector3
    gravitational_force_n: Vector3
    electric_potential_energy_j: float
    gravitational_potential_energy_j: float

    def __post_init__(self):
        values = (*self.electric_force_n, *self.gravitational_force_n,
                  self.electric_potential_energy_j, self.gravitational_potential_energy_j)
        if not all(isfinite(x) for x in values):
            raise ValueError('nonfinite conservative force/energy contribution')
        if not all(isfinite(x) for x in (*self.force_n, self.potential_energy_j)):
            raise ValueError('nonfinite conservative force/energy sum')

    @property
    def force_n(self):
        return tuple(fsum((a, b)) for a, b in zip(self.electric_force_n, self.gravitational_force_n))

    @property
    def potential_energy_j(self):
        return fsum((self.electric_potential_energy_j, self.gravitational_potential_energy_j))

    @property
    def energy(self):
        # Existing integrator already differences contributions independently.
        # No induced term: prescribed electric energy is direct qV and gravity
        # is the conservative mechanical background. No half factor is used.
        return EnergyBreakdown(self.electric_potential_energy_j, 0.,
                               self.gravitational_potential_energy_j, COMPOSED_ENERGY)


@dataclass(frozen=True, slots=True)
class ConservativeForceSystem:
    """Two qualified built-in terms, not an unrestricted force plugin registry."""
    electric: PrescribedElectricForce
    gravity: UniformGravity | None = None

    def __post_init__(self):
        if type(self.electric) is not PrescribedElectricForce:
            raise TypeError('require the built-in prescribed electric force component')
        if self.gravity is not None and type(self.gravity) is not UniformGravity:
            raise TypeError('require built-in UniformGravity or None')

    def observe(self, position_m, *, charge_c, mass_kg):
        _number(charge_c, 'particle charge')
        _number(mass_kg, 'particle mass', positive=True)
        gravity = self.gravity
        try:
            return ForceObservation(
                self.electric.force_at(position_m, charge_c=charge_c),
                gravity.force_at(position_m, mass_kg=mass_kg) if gravity else (0., 0., 0.),
                self.electric.potential_energy_at(position_m, charge_c=charge_c),
                gravity.potential_energy_at(position_m, mass_kg=mass_kg) if gravity else 0.)
        except (OverflowError, ValueError) as exc:
            raise ValueError('nonfinite or invalid conservative force observation') from exc


def scalar_matches(actual, expected, *, scale=0.):
    """Scale-aware verification down to proton energies; no macroscopic floor.

    The absolute allowance is floating-point roundoff of the physical terms,
    including when their resultant cancels. It is not a force cutoff.
    """
    return (type(actual) in (int, float) and isfinite(actual) and
            isclose(actual, expected, rel_tol=3e-9,
                    abs_tol=128*ulp(max(abs(expected), abs(scale)))))


def validate_contributions(record, expected):
    """Verify saved independent contributions, not merely their cancelling sum."""
    for name in CONTRIBUTION_KEYS:
        actual, value = record.get(name), getattr(expected, name)
        if isinstance(value, tuple):
            if (not isinstance(actual, (tuple, list)) or len(actual) != 3 or
                    any(not scalar_matches(a, b, scale=hypot(*value)) for a, b in zip(actual, value))):
                raise ValueError(f'saved {name} contradicts its physical component')
        elif not scalar_matches(actual, value):
            raise ValueError(f'saved {name} contradicts its physical component')


def validate_composed_sample(spec, record):
    """Shared result/export verification with physical-scale error allowances."""
    obs = spec.forces.observe(record['position_m'], charge_c=spec.particle.charge_c,
                              mass_kg=spec.mass_kg)
    initial = spec.forces.observe(spec.particle.position_m, charge_c=spec.particle.charge_c,
                                  mass_kg=spec.mass_kg)
    validate_contributions(record, obs)
    force_scale = max(hypot(*obs.electric_force_n), hypot(*obs.gravitational_force_n))
    for name, expected, scale in (
        ('force_n', obs.force_n, force_scale),
        ('acceleration_m_s2', tuple(f/spec.mass_kg for f in obs.force_n), force_scale/spec.mass_kg),
        ('electric_field_v_m', spec.field.electric_field_at(record['position_m']), 0.)):
        if any(not scalar_matches(a, b, scale=scale) for a, b in zip(record[name], expected)):
            raise ValueError(f'saved {name} violates the composed physical force model')
    if not scalar_matches(record['potential_v'], spec.field.potential_at(record['position_m'])):
        raise ValueError('electric potential must remain the actual electric V, not gravity')
    K = .5*spec.mass_kg*hypot(*record['velocity_m_s'])**2
    K0 = .5*spec.mass_kg*hypot(*spec.initial_velocity_m_s)**2
    scale = max(abs(K), abs(K0), abs(obs.electric_potential_energy_j),
                abs(obs.gravitational_potential_energy_j),
                abs(initial.electric_potential_energy_j), abs(initial.gravitational_potential_energy_j))
    drift = fsum((K-K0, obs.electric_potential_energy_j-initial.electric_potential_energy_j,
                  obs.gravitational_potential_energy_j-initial.gravitational_potential_energy_j))
    for name, expected in (
        ('kinetic_energy_j', K), ('potential_energy_j', obs.potential_energy_j),
        ('total_energy_j', K+obs.potential_energy_j), ('energy_drift_j', drift)):
        if not scalar_matches(record[name], expected, scale=scale):
            raise ValueError('composed energy must be K+qV+Ug with separate contribution drift')
    if 'interaction_energy_j' in record and record['interaction_energy_j'] != record['potential_energy_j']:
        raise ValueError('shared viewer U must equal complete qV+Ug, not just qV')
