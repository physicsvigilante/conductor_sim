"""Phase 0c: exterior-sphere scale invariance (M01 C9/F3) and intended-surface sigma (C10/F4).

The oracle is a 60-digit Decimal Kelvin construction written from the closed
form (image -q*a/d at a**2/d along the source direction, plus the central
monopole Q - q_image for a fixed-net-charge sphere). It never calls the code
under test. Inputs are the exact binary values of the authored floats.
"""
from decimal import Decimal as D, localcontext
from math import isclose, isfinite, pi

import pytest

from conductor_sim import (
    EPSILON_0, ElectrostaticProblem, ExternalGroundedSphereBackend, ExternalIsolatedSphereBackend,
    ExternalMultiSphereBackend, FixedNetCharge, Grounded, PointCharge, SolidSphere,
)

SCALES = (1e-150, 1.0, 1e150, 1e160)


def _dec(v):
    return tuple(D.from_float(float(x)) for x in v)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _len(v):
    return sum(x*x for x in v).sqrt()


class Kelvin:
    """Exact-arithmetic Kelvin reference for point sources outside one sphere at the origin."""

    def __init__(self, a, sources, net=None):
        self.a = D.from_float(a)
        self.k = 1/(4*D(pi)*D.from_float(EPSILON_0))
        self.real = [(D.from_float(q), _dec(p)) for q, p in sources]
        self.images = []
        for q, p in self.real:
            d = _len(p)
            self.images.append((-q*self.a/d, tuple(x*self.a*self.a/(d*d) for x in p)))
        total_image = sum(q for q, _ in self.images)
        self.central = D(0) if net is None else D.from_float(net)-total_image
        self.all = self.real+self.images+[(self.central, (D(0), D(0), D(0)))]

    def potential(self, point):
        p = _dec(point)
        return self.k*sum(q/_len(_sub(p, s)) for q, s in self.all if q)

    def field(self, point):
        p = _dec(point)
        out = [D(0)]*3
        for q, s in self.all:
            if q:
                v = _sub(p, s)
                r = _len(v)
                for i in range(3):
                    out[i] += self.k*q*v[i]/(r*r*r)
        return out

    def sigma(self, point):
        # Density at the radial projection of the query onto r = a.
        p = _dec(point)
        u = tuple(x/_len(p) for x in p)
        total = self.central/(4*D(pi)*self.a*self.a)
        for q, s in self.real:
            d = _len(s)
            length = _len(_sub(tuple(self.a*x for x in u), s))
            total += -q*(d*d-self.a*self.a)/(4*D(pi)*self.a*length**3)
        return total

    def force(self, index):
        q, p = self.real[index]
        out = [D(0)]*3
        for j, (qs, s) in enumerate(self.all):
            if qs and not (j < len(self.real) and j == index):
                v = _sub(p, s)
                r = _len(v)
                for i in range(3):
                    out[i] += self.k*q*qs*v[i]/(r*r*r)
        return out


def _close_vector(actual, exact, rel=1e-12):
    error = sum((D.from_float(x)-y)**2 for x, y in zip(actual, exact)).sqrt()
    size = sum(y*y for y in exact).sqrt()
    assert all(isfinite(x) for x in actual) and error <= D(rel)*size, (actual, exact)


def _close(actual, exact, rel=1e-12):
    assert isfinite(actual) and abs(D.from_float(actual)-exact) <= D(rel)*abs(exact), (actual, exact)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("isolated", (False, True))
def test_single_exterior_source_is_scale_invariant(scale, isolated):
    # W-c1 (F3): charge per unit length fixed, so V is scale invariant.
    a, q = scale, 1e-9*scale
    net = -.25*q if isolated else None
    sphere = SolidSphere(a, FixedNetCharge(net) if isolated else Grounded())
    problem = ElectrostaticProblem((PointCharge(q, (2*a, 0., 0.)),), (sphere,))
    backend = ExternalIsolatedSphereBackend() if isolated else ExternalGroundedSphereBackend()
    assert backend.supports(problem)
    solution = backend.solve(problem)
    sample = (3*a, .5*a, .25*a)
    surface = (.6*a, .8*a, 0.)
    with localcontext() as context:
        context.prec = 60
        ref = Kelvin(a, ((q, (2*a, 0., 0.)),), net)
        assert all(isfinite(x) for x in solution.image_position_m)
        _close(solution.potential_at(sample), ref.potential(sample))
        _close_vector(solution.electric_field_at(sample), ref.field(sample))
        _close(solution.surface_charge_density_at("outer", surface), ref.sigma(surface))
        _close_vector(solution.induced_force_on_charge_n, ref.force(0))
        if isolated:
            # The grounded Kelvin pair vanishes on r=a; the wall potential is
            # that of the central monopole alone.
            _close(solution.conductor_potential_v, ref.k*ref.central/ref.a)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("isolated", (False, True))
def test_multi_exterior_sources_are_scale_invariant(scale, isolated):
    # W-c2: the same scaling for the multi-source Kelvin superposition.
    a, q = scale, 1e-9*scale
    sources = ((q, (2*a, 0., 0.)), (-.5*q, (0., 3*a, a)))
    net = .75*q if isolated else None
    sphere = SolidSphere(a, FixedNetCharge(net) if isolated else Grounded())
    problem = ElectrostaticProblem(tuple(PointCharge(c, p) for c, p in sources), (sphere,))
    solution = ExternalMultiSphereBackend().solve(problem)
    sample = (-2*a, a, .5*a)
    surface = (0., .6*a, .8*a)
    with localcontext() as context:
        context.prec = 60
        ref = Kelvin(a, sources, net)
        site = next(i for i, s in enumerate(solution.sites) if s.position_m == (2*a, 0., 0.))
        _close(solution.potential_at(sample), ref.potential(sample))
        _close_vector(solution.electric_field_at(sample), ref.field(sample))
        _close(solution.surface_charge_density_at("outer", surface), ref.sigma(surface))
        _close_vector(solution.force_on_charge_n(site), ref.force(0))


def test_multi_sigma_uses_intended_surface_point_near_contact():
    # W-c3 (F4): admitted off-radius query 5e-11*a inside the surface, source
    # 1e-8*a outside it. The physical pole density is -q(d+a)/(4 pi a (d-a)**2).
    a, q = .5, -3e-9
    d = a*(1+1e-8)
    half = PointCharge(q/2, (d, 0., 0.))
    solution = ExternalMultiSphereBackend().solve(
        ElectrostaticProblem((half, half), (SolidSphere(a, Grounded()),)))
    with localcontext() as context:
        context.prec = 80
        aa, dd, qq = D.from_float(a), D.from_float(d), D.from_float(q)
        pole = -qq*(dd+aa)/(4*D(pi)*aa*(dd-aa)**2)
        _close(solution.surface_charge_density_at("outer", (a*(1-5e-11), 0., 0.)), pole)
        _close(solution.surface_charge_density_at("outer", (a, 0., 0.)), pole)
        ref = Kelvin(a, ((q, (d, 0., 0.)),))
        for point in ((a*(1+3e-11), a*1e-7, 0.), (a*.6, a*.8*(1-4e-11), 0.)):
            _close(solution.surface_charge_density_at("outer", point), ref.sigma(point))


def test_single_sigma_near_contact_off_axis():
    # Same intended-surface contract on the single-source grounded/isolated path.
    a, q = .5, -3e-9
    d = a*(1+1e-8)
    point = (a*(1-4e-11), a*2e-8, 0.)
    for constraint, net in ((Grounded(), None), (FixedNetCharge(1e-9), 1e-9)):
        problem = ElectrostaticProblem((PointCharge(q, (d, 0., 0.)),), (SolidSphere(a, constraint),))
        backend = ExternalGroundedSphereBackend() if net is None else ExternalIsolatedSphereBackend()
        solution = backend.solve(problem)
        with localcontext() as context:
            context.prec = 80
            _close(solution.surface_charge_density_at("outer", point),
                   Kelvin(a, ((q, (d, 0., 0.)),), net).sigma(point))


def test_ordinary_values_unchanged_within_rounding():
    # Compatibility at an ordinary JEE scale: angular sigma API and isolated
    # energy agree with the closed forms to rounding.
    a, q, d, net = .1, 2e-9, .3, -1e-9
    problem = ElectrostaticProblem((PointCharge(q, (d, 0., 0.)),), (SolidSphere(a, FixedNetCharge(net)),))
    solution = ExternalIsolatedSphereBackend().solve(problem)
    k = 1/(4*pi*EPSILON_0)
    b = a*a/d
    qi = -q*a/d
    energy = .5*k*q*qi/(d-b) + k*q*net/d + k*q*q*a/(2*d*d)
    assert isclose(solution.interaction_energy_j, energy, rel_tol=1e-14)
    sigma0 = -q*(d*d-a*a)/(4*pi*a*(d-a)**3) + (net-qi)/(4*pi*a*a)
    assert isclose(solution.surface_charge_density("outer", 0.), sigma0, rel_tol=1e-14)


EXTREME_MULTI = (
    # (a, q, constraint net or None): forces and energies are normal floats
    # even though q/r**2 (first case) or k*q (third case) is not.
    (1e-260, 1e-200, None),
    (1e270, 1e200, None),
    (1e300, 1e299, -.25),
)


@pytest.mark.parametrize("a, q, net_ratio", EXTREME_MULTI)
def test_multi_forces_and_energy_at_extreme_charge_scales(a, q, net_ratio):
    # Review 0c-B1: each pair force is (k*(q/r))*(q'/r)*(v/r); no k*q or q'/r**2.
    sources = ((q, (2*a, 0., 0.)), (-.5*q, (0., 3*a, 0.)))
    net = None if net_ratio is None else net_ratio*q
    sphere = SolidSphere(a, Grounded() if net is None else FixedNetCharge(net))
    solution = ExternalMultiSphereBackend().solve(
        ElectrostaticProblem(tuple(PointCharge(c, p) for c, p in sources), (sphere,)))
    site = next(i for i, s in enumerate(solution.sites) if s.position_m == (2*a, 0., 0.))
    with localcontext() as context:
        context.prec = 60
        ref = Kelvin(a, sources, net)
        _close_vector(solution.force_on_charge_n(site), ref.force(0))
        p = ref.real[0][1]
        metal = [D(0)]*3
        for qs, s in ref.images+[(ref.central, (D(0), D(0), D(0)))]:
            if qs:
                v = _sub(p, s)
                r = _len(v)
                for i in range(3):
                    metal[i] += ref.k*ref.real[0][0]*qs*v[i]/(r*r*r)
        _close_vector(solution.conductor_force_on_charge_n(site), metal)
        _close(solution.potential_at((3*a, a, 0.)), ref.potential((3*a, a, 0.)))
        if net is not None:
            _close(solution.conductor_potential_v, ref.k*ref.central/ref.a)


@pytest.mark.parametrize("a, d", ((1.0, 1e200), (1e100, 1e270), (1e-100, 1e60)))
def test_distant_source_image_position_is_normal(a, d):
    # Review 0c-B2: image radius a*(a/d) along delta/d, never (a/d)**2*delta.
    q = d
    half = PointCharge(q/2, (d, 0., 0.))
    multi = ExternalMultiSphereBackend().solve(
        ElectrostaticProblem((half, half), (SolidSphere(a, Grounded()),)))
    single = ExternalGroundedSphereBackend().solve(
        ElectrostaticProblem((PointCharge(q, (d, 0., 0.)),), (SolidSphere(a, Grounded()),)))
    with localcontext() as context:
        context.prec = 60
        exact = D.from_float(a)*D.from_float(a)/D.from_float(d)
        _close(multi.image_sources[0].position_m[0], exact)
        _close(single.image_position_m[0], exact)


def test_isolated_potentials_when_k_times_charge_overflows():
    # Review 0c-M1: k*(Q_c/a), not (k*Q_c)/a.
    a, q = 1e300, 1e299
    net = -.25*q
    solution = ExternalIsolatedSphereBackend().solve(ElectrostaticProblem(
        (PointCharge(q, (2*a, 0., 0.)),), (SolidSphere(a, FixedNetCharge(net)),)))
    with localcontext() as context:
        context.prec = 60
        ref = Kelvin(a, ((q, (2*a, 0., 0.)),), net)
        wall = ref.k*ref.central/ref.a
        _close(solution.conductor_potential_v, wall)
        _close(solution.conductor_results[0].potential_v, wall)
        _close(solution.potential_at((3*a, 0., 0.)), ref.potential((3*a, 0., 0.)))
        _close_vector(solution.induced_force_on_charge_n, ref.force(0))


@pytest.mark.parametrize("scale", SCALES)
def test_interaction_energy_at_extreme_scales(scale):
    # Energies at the W-c1 scales against the closed forms in exact arithmetic.
    a, q = scale, 1e-9*scale
    with localcontext() as context:
        context.prec = 60
        k = 1/(4*D(pi)*D.from_float(EPSILON_0))
        aa, qq, dd = D.from_float(a), D.from_float(q), D.from_float(2*a)
        qi, b = -qq*aa/dd, aa*aa/dd
        grounded = k*qq*qi/(2*(dd-b))
        g = ExternalGroundedSphereBackend().solve(ElectrostaticProblem(
            (PointCharge(q, (2*a, 0., 0.)),), (SolidSphere(a, Grounded()),)))
        _close(g.interaction_energy_j, grounded)
        net = -.25*q
        QQ = D.from_float(net)
        isolated = grounded + k*qq*QQ/dd + k*qq*qq*aa/(2*dd*dd)
        i = ExternalIsolatedSphereBackend().solve(ElectrostaticProblem(
            (PointCharge(q, (2*a, 0., 0.)),), (SolidSphere(a, FixedNetCharge(net)),)))
        _close(i.interaction_energy_j, isolated)
