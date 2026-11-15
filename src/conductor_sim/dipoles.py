"""Finite rigid dipoles in existing nonresponding external fields, SI.

Static observations only. No inertia, elapsed time, source response or induced
polarization. External interaction energy excludes constant mutual binding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
import json
from math import cos, fsum, hypot, isfinite, sin, ulp
import re

from .model import EPSILON_0
from .prescribed_fields import (PrescribedField, UniformElectricField,
    _dot, _mul, _number, _sub, _unit, _vector)
from .solver import SingularField, UnsupportedProblem
from .structural_scenarios import NamedSource


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _sum(vectors):
    return tuple(fsum(v[i] for v in vectors) for i in range(3))


def _native(x):
    return json.loads(json.dumps(x, allow_nan=False))


def _id(value):
    if type(value) is not str or re.fullmatch(r'[A-Za-z][A-Za-z0-9:_-]{0,63}', value) is None:
        raise ValueError('physical identity must be 1..64 identifier characters')
    return value


@dataclass(frozen=True, slots=True)
class RigidDipole:
    """Direction points from -q to +q. Optional pivot is fixed in world space.

    A pivot specifies a static rotation investigation, NOT a simulated joint.
    With pivot_m=None, requested orientations rotate about the fixed centre.
    """
    charge_c: float
    separation_m: float
    center_m: tuple = (0., 0., 0.)
    direction: tuple = (1., 0., 0.)
    pivot_m: tuple | None = None
    source_id: str = 'dipole'

    def __post_init__(self):
        for name in ('charge_c', 'separation_m'):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        object.__setattr__(self, 'center_m', _vector(self.center_m, 'dipole centre'))
        object.__setattr__(self, 'direction', _unit(self.direction, 'dipole direction'))
        if self.pivot_m is not None:
            object.__setattr__(self, 'pivot_m', _vector(self.pivot_m, 'pivot'))
        _id(self.source_id)
        ends = self.endpoints_m
        resolution = max(ulp(v) for p in (*ends, self.torque_origin_m) for v in p)
        if (not all(isfinite(v) for p in ends for v in p) or
                64*resolution > 1e-10*self.separation_m or
                not isfinite(self.charge_c*self.separation_m) or
                self.charge_c*self.separation_m == 0.):
            raise UnsupportedProblem('dipole geometry or moment is poorly resolved; rescale coordinates')

    @property
    def endpoints_m(self):
        half = _mul(self.direction, self.separation_m/2)
        return _sum((self.center_m, half)), _sub(self.center_m, half)

    @property
    def moment_c_m(self):
        return _mul(self.direction, self.charge_c*self.separation_m)

    @property
    def torque_origin_m(self):
        return self.center_m if self.pivot_m is None else self.pivot_m

    def as_dict(self):
        return _native(asdict(self))


@dataclass(frozen=True, slots=True)
class DipoleModel:
    """Named existing field components plus one rigid probe, no mutual response."""
    dipole: RigidDipole
    sources: tuple
    model_id: str
    provenance: str
    permittivity_f_m: float = EPSILON_0

    def __post_init__(self):
        if type(self.dipole) is not RigidDipole:
            raise TypeError('expected RigidDipole')
        if type(self.sources) not in (list, tuple) or any(type(s) is not NamedSource for s in self.sources):
            raise TypeError('external inventory must contain existing NamedSource components')
        object.__setattr__(self, 'sources', tuple(self.sources))
        ids = [s.source_id for s in self.sources]
        if len(set(ids)) != len(ids) or self.dipole.source_id in ids:
            raise ValueError('duplicate identity or probe included in its external inventory')
        _id(self.model_id)
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError('explicit source/teacher provenance is required')
        self.field  # Validate the existing field regime, without requesting observations.

    @property
    def field(self):
        return PrescribedField(tuple(s.source for s in self.sources), self.permittivity_f_m)

    def as_dict(self):
        return _native(dict(schema='conductor_sim.dipole_model.v1', model_id=self.model_id,
                    provenance=self.provenance, dipole=self.dipole.as_dict(),
                    sources=[s.as_dict() for s in self.sources],
                    permittivity_f_m=self.permittivity_f_m))

    @property
    def digest(self):
        return sha256(json.dumps(self.as_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()


def observe_dipole(model: DipoleModel):
    """Exact finite-pair external U, F and torque; no point-dipole approximation.

    Finite potential at a surface still permits U when E/torque is undefined.
    Unsupported physics and singular potentials remain explicit errors.
    """
    if type(model) is not DipoleModel:
        raise TypeError('expected DipoleModel')
    d = model.dipole
    plus, minus = d.endpoints_m
    parts = []
    for named in model.sources:
        source = named.source
        vp = source.potential_at(plus, permittivity_f_m=model.permittivity_f_m)
        vm = source.potential_at(minus, permittivity_f_m=model.permittivity_f_m)
        # Exact neutral-pair difference cancels an arbitrary uniform-field gauge
        # algebraically, avoiding subtraction of two huge equal gauge constants.
        energy = (-_dot(d.moment_c_m, source.field_v_m) if type(source) is UniformElectricField
                  else d.charge_c*(vp-vm))
        forces, reasons = [], []
        for p, q in ((plus, d.charge_c), (minus, -d.charge_c)):
            try:
                forces.append(_mul(source.electric_field_at(p,
                    permittivity_f_m=model.permittivity_f_m), q))
                reasons.append(None)
            except SingularField as exc:
                forces.append(None)
                reasons.append(str(exc))
        parts.append(dict(source_id=named.source_id, label=named.label,
                          interaction_energy_j=energy, endpoint_force_n=forces,
                          endpoint_field_unavailable=reasons))
    force = [None if any(p['endpoint_force_n'][i] is None for p in parts)
             else _sum([p['endpoint_force_n'][i] for p in parts]) for i in (0, 1)]
    complete = all(f is not None for f in force)
    net = _sum(force) if complete else None
    torque = (_sum([_cross(_sub(p, d.torque_origin_m), f)
                    for p, f in zip((plus, minus), force)]) if complete else None)
    result = dict(schema='conductor_sim.dipole_observation.v1', model_digest=model.digest,
        mode='static configuration; no elapsed time', dipole=d.as_dict(),
        positive_position_m=plus, negative_position_m=minus,
        dipole_moment_c_m=d.moment_c_m, torque_origin_m=d.torque_origin_m,
        positive_force_n=force[0], negative_force_n=force[1],
        net_force_n=net, torque_n_m=torque,
        interaction_energy_j=fsum(p['interaction_energy_j'] for p in parts),
        source_contributions=parts,
        energy_convention='External interaction q(V_plus-V_minus); fixed-length binding energy excluded')
    return _native(result)  # Reject overflow/nonfinite observations, never silently clip.


def orient_dipole(model, angle_rad, *, rotation_axis=(0., 0., 1.)):
    """New static pose relative to the supplied model, including pivot-centre motion."""
    a = _number(angle_rad, 'orientation angle')
    n = _unit(rotation_axis, 'rotation axis')
    def rotate(v):
        return _sum((_mul(v, cos(a)), _mul(_cross(n, v), sin(a)),
                     _mul(n, _dot(n, v)*(1-cos(a)))))
    d = model.dipole
    centre = _sum((d.torque_origin_m, rotate(_sub(d.center_m, d.torque_origin_m))))
    return replace(model, dipole=replace(d, center_m=centre, direction=rotate(d.direction)))


def dipole_work(model, start_angle_rad, end_angle_rad, *, rotation_axis=(0., 0., 1.)):
    """Endpoint energy difference for one unchanged model and pivot policy.

    No admissible continuous path, dissipated energy or transit time is claimed.
    """
    a = observe_dipole(orient_dipole(model, start_angle_rad, rotation_axis=rotation_axis))
    b = observe_dipole(orient_dipole(model, end_angle_rad, rotation_axis=rotation_axis))
    change = b['interaction_energy_j']-a['interaction_energy_j']
    return dict(external_quasistatic_work_j=change, electric_work_j=-change,
                interpretation='Endpoint difference only; a continuous rotation path is not certified')


def sample_orientations(model, angles_rad, *, rotation_axis=(0., 0., 1.)):
    """Request independent static orientations; refuse an invalid field regime.

    Singular potential samples are explicit gaps. Undefined forces do not
    erase available energy. Samples do not establish intervening admissibility.
    """
    if type(angles_rad) not in (list, tuple) or not 2 <= len(angles_rad) <= 361:
        raise ValueError('supply 2..361 explicit angles')
    angles = tuple(_number(a, 'angle') for a in angles_rad)
    if any(b <= a for a, b in zip(angles, angles[1:])):
        raise ValueError('angles must be strictly increasing')
    n = _unit(rotation_axis, 'rotation axis')
    rows = []
    for a in angles:
        pose = orient_dipole(model, a, rotation_axis=n)
        try:
            obs = observe_dipole(pose)
            row = dict(angle_rad=a, observation=obs, status='observed')
        except SingularField as exc:
            row = dict(angle_rad=a, observation=None, status='singular-potential', reason=str(exc))
        rows.append(row)
    return dict(schema='conductor_sim.dipole_orientation_samples.v1',
        model=model.as_dict(), model_digest=model.digest, rotation_axis=list(n), rows=rows,
        mode='Independent static poses, not time integration or continuous-path certification')


def uniform_rotational_equilibria(model):
    """Exact rotational classification ONLY for globally uniform external fields.

    Does not assert isolated translational equilibrium or damping/alignment time.
    """
    if any(type(s.source) is not UniformElectricField for s in model.sources):
        raise UnsupportedProblem('nonuniform equilibrium search is not qualified in M23a')
    e = _sum([s.source.field_v_m for s in model.sources])
    magnitude = hypot(*e)
    if not isfinite(magnitude):
        raise ValueError('uniform field overflow')
    if magnitude == 0.:
        return dict(classification='neutral: all orientations', orientations=[])
    n = _unit(e, 'resultant uniform field')
    return dict(classification='uniform-field rotational equilibria', orientations=[
        dict(direction=list(n), stability='stable rotational minimum'),
        dict(direction=list(_mul(n, -1)), stability='unstable rotational maximum')])


def revise_dipole_model(parent, *, model_id, note, dipole_changes=None, source_changes=None):
    """Bounded immutable property revision; membership and source kinds stay fixed."""
    if model_id == parent.model_id or type(note) is not str or not note.strip():
        raise ValueError('revision requires a new model identity and explanatory note')
    dc = {} if dipole_changes is None else dipole_changes
    sc = {} if source_changes is None else source_changes
    if type(dc) is not dict or type(sc) is not dict:
        raise TypeError('changes must be dictionaries')
    allowed = {f.name for f in fields(RigidDipole)}-{'source_id'}
    if set(dc)-allowed or set(sc)-{s.source_id for s in parent.sources}:
        raise ValueError('unknown or identity-changing property revision')
    sources = []
    for named in parent.sources:
        patch = sc.get(named.source_id, {})
        if type(patch) is not dict or set(patch)-{f.name for f in fields(named.source)}:
            raise ValueError('unsupported source property')
        sources.append(replace(named, source=replace(named.source, **patch)))
    child = DipoleModel(replace(parent.dipole, **dc), tuple(sources), model_id,
                        'Teacher hypothetical: '+note, parent.permittivity_f_m)
    # No analysis is automatically requested here: force, work and orientation
    # qualification are separate from model authoring.
    before, after = parent.as_dict(), child.as_dict()
    changes = []
    for key in sorted(allowed):
        if before['dipole'][key] != after['dipole'][key]:
            changes.append(dict(target=parent.dipole.source_id, property=key,
                before=before['dipole'][key], after=after['dipole'][key]))
    for old, new in zip(before['sources'], after['sources']):
        for key in sorted(old['source']):
            if old['source'][key] != new['source'][key]:
                changes.append(dict(target=old['id'], property=key,
                    before=old['source'][key], after=new['source'][key]))
    return child, dict(schema='conductor_sim.dipole_revision.v1',
        parent_digest=parent.digest, child_digest=child.digest,
        parent=before, child=after, actual_changes=changes)
