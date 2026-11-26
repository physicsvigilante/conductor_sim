"""Analytical reference for two infinite parallel conducting half-spaces.

Both surface normals must face *into the same open slab*, so the authored
vacuum is the intersection of their positive-normal half-spaces. A linear
Dirichlet background supports arbitrary prescribed plate potentials. Point
sources use a finite *reciprocal, plate-symmetric* image-charge series.
Admitted local observations (``potential_at``, ``electric_field_at``,
``surface_charge_density_at``) are certified within 1 % and sign-correct
under all plate biases; uncertified samples raise ``OutsideDomain``.

Rigorous finite-N remainder bound derivation (R-b1, zero fitted constants)
--------------------------------------------------------------------------
For one active point charge ``q`` at height ``z0 = u0 * d`` in ``(0, d)``,
query height ``h = u * d`` in ``[0, d]``, and transverse offset ``rho >= 0``,
set ``a = h - z0``, ``b = h + z0``, ``alpha = (u + u0) / 2 in (0, 1)``,
``beta = (u0 - u) / 2 in (-0.5, 0.5)``, and ``r(rho, L) = hypot(rho, L)``.
Let ``K(L)`` denote the kernel ``f0(L) = 1/r`` (potential, even),
``g(L) = L/r**3`` (normal field, odd), or ``w(L) = rho/r**3`` (radial field,
even). Subtracting the finite averaged sum ``S_N`` (positive ``n = -N..N``;
negative half-weighted ``-N..N`` and ``1-N..N+1``) from the exact symmetric
limit ``S_inf`` gives the exact algebraic identity:

    R_N = S_inf - S_N = E_end + sum_{n=N+1}^inf T_n

where, with ``A_N = 2(N+1)d - b > 2Nd``, ``B_N = 2Nd + b > 2Nd``, and
``L_min = min(A_N, B_N) > 2Nd``:

    E_end = (1/2) * (K(b - 2(N+1)d) - K(b + 2Nd))
    T_n   = K(a - 2nd) + K(a + 2nd) - K(b - 2nd) - K(b + 2nd).

Universal uniform-in-rho derivative bounds (valid for all L > 0, rho >= 0):
  - f0 (even): |f0'(L)| <= 1/L**2;
               |f0''(L)| = |2L**2 - rho**2|/r**5 <= 2/r**3 <= 2/L**3.
  - g  (odd):  |g(L)| <= 1/L**2;
               |g'(L)| = |rho**2 - 2L**2|/r**5 <= 2/r**3 <= 2/L**3.
  - w  (even): |w'(L)| = 3*rho*L/r**5 <= min(1.5/L**3, 3*rho/L**4);
               |w''(L)| = 3*rho*|4L**2 - rho**2|/r**7
                        <= min((16/sqrt(15))/L**4, 12*rho/L**5)
               (exact max of |3s(4 - 5s**2)| on s in [0, 1] is 16/sqrt(15)).

Integral-test tail bound for N >= 1, eta < 1, p >= 2:
    I_p(N, eta) = (N + 1 - eta)**(-p) + (N + 1 - eta)**(1 - p) / (p - 1)
                >= sum_{n=N+1}^inf (n - eta)**(-p).

Applying the Mean Value Theorem to ``E_end`` and the exact FTC double/single
integral identities to ``T_n``:
  - Potential (f0 even):
      |E_end,V| <= |b - d| / L_min**2
      |T_{n,V}| = |int_{|a|}^b int_{-t}^t f0''(2nd + s) ds dt|
                <= 2*(b**2 - a**2)/(2nd - b)**3 = 8*h*z0/(2d*(n - alpha))**3
      => sum |T_{n,V}| <= (u * u0 / d) * I_3(N, alpha).
  - Normal field & sigma (g odd):
      |E_end,En| <= 0.5 * (1/A_N**2 + 1/B_N**2)
      |T_{n,En}| = |int_a^b g'(2nd + s) ds + int_{-b}^{-a} g'(2nd + s) ds|
                 <= (u0 / (2 * d**2)) * ((n - alpha)**(-3) + (n - beta)**(-3))
      => sum |T_{n,En}| <= (u0 / (2 * d**2)) * (I_3(N, alpha) + I_3(N, beta)).
  - Radial field (w even):
      |E_end,Er| <= |b - d| * min(1.5/L_min**3, 3*rho/L_min**4)
      sum |T_{n,Er}| <= min((u * u0 * (4/sqrt(15)) / d**2) * I_4(N, alpha),
                            (1.5 * u * u0 * rho / d**3) * I_5(N, alpha)).

Plate-swap symmetry: ``_image_terms`` maps ``(z1, z2, z3)`` to
``(-z1(-n), -z3(-n), -z2(-n))`` under ``(u, u0) -> (1 - u, 1 - u0)``, so
``|R_N(u, u0)| == |R_N(1 - u, 1 - u0)|`` identically; taking
``min(B(u, u0), B(1 - u, 1 - u0))`` is exact.

Representational floor: ``math.fsum`` rounds the list of terms to <= 0.5 ulp,
so floating-point error is bounded by Higham's forward-error bound
``16 * u_mach`` (``u_mach = 2**-53``) times the sum of positive per-term
envelopes plus ``4 * u_mach`` times the plate-bias scale.

Admission criterion (all plate biases, t = 0.01): admit iff ``|x_hat| > B``
and ``B <= t * (|x_hat| - B)``, guaranteeing ``|x_true| >= |x_hat| - B > 0``
(exact sign) and ``|x_hat - x_true| / |x_true| <= t``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, hypot, isclose, isfinite, pi, sqrt

from .grounded_plane import _sites, _dot, _minus
from .model import (
    ConductingPlane, ElectrostaticProblem, FixedPotential, Grounded,
    PointCharge, Vector3, _position,
)
from .solver import ConductorResult, OutsideDomain, SingularField, UnsupportedProblem

_CERT_REL_TOL = 0.01
_U_MACH = 2.0 ** -53
_C_W2 = 16.0 / sqrt(15.0)


def _voltage(plate: ConductingPlane) -> float:
    return 0.0 if isinstance(plate.constraint, Grounded) else plate.constraint.potential_v


def _int_sum_inv_power(order: int, eta: float, power: int) -> float:
    """Upper bound on sum_{n=order+1}^inf (n - eta)**(-power) for order>=1, eta<1, power>=2."""
    m = order + 1.0 - eta
    return (m ** -power) + (m ** (1 - power)) / (power - 1.0)


def _tail_bounds_orientation(d: float, order: int, u: float, u0: float, rho: float) -> tuple[float, float, float]:
    """Dimensionless (divided by k*|q|) rigorous truncation bounds for V, E_n, E_rho."""
    b = (u + u0) * d
    alpha = 0.5 * (u + u0)
    beta = 0.5 * (u0 - u)
    a_n = 2.0 * (order + 1.0) * d - b
    b_n = 2.0 * order * d + b
    l_min = min(a_n, b_n)

    # Potential V (f0 even)
    e_end_v = abs(b - d) / (l_min * l_min)
    tail_v = (u * u0 / d) * _int_sum_inv_power(order, alpha, 3)

    # Normal field E_n and sigma (g odd)
    e_end_en = 0.5 * (1.0 / (a_n * a_n) + 1.0 / (b_n * b_n))
    tail_en = (0.5 * u0 / (d * d)) * (
        _int_sum_inv_power(order, alpha, 3) + _int_sum_inv_power(order, beta, 3)
    )

    # Radial field E_rho (w even)
    e_end_er = abs(b - d) * min(1.5 / (l_min ** 3), 3.0 * rho / (l_min ** 4))
    tail_er_unif = (0.25 * u * u0 * _C_W2 / (d * d)) * _int_sum_inv_power(order, alpha, 4)
    tail_er_small_rho = (1.5 * u * u0 * rho / (d ** 3)) * _int_sum_inv_power(order, alpha, 5)

    return e_end_v + tail_v, e_end_en + tail_en, e_end_er + min(tail_er_unif, tail_er_small_rho)


def _certify_total(observable: str, observed: float, bound: float, image_order: int, *,
                   t: float = _CERT_REL_TOL) -> None:
    """Enforce |observed| > bound and bound <= t * (|observed| - bound) under all biases."""
    if bound == 0.0:
        return
    mag = abs(observed)
    if mag <= bound or bound > t * (mag - bound):
        raise OutsideDomain(
            f"parallel-plate {observable} is uncertified at image_order={image_order}: "
            f"observed={observed:.6e}, bound={bound:.6e}, rel_tol={t}"
        )


def _layout(problem: ElectrostaticProblem):
    if (len(problem.conductors) != 2 or
            any(not isinstance(c, ConductingPlane) for c in problem.conductors)):
        raise UnsupportedProblem("parallel plates require exactly two ConductingPlane objects")
    a, b = problem.conductors
    # Opposing normals are necessary: two equal-facing surfaces do not bound
    # an open slab, and a tilted pair is not an infinite parallel capacitor.
    if hypot(*(a.normal[i]+b.normal[i] for i in range(3))) > 1e-15:
        raise UnsupportedProblem("plate normals must be parallel, opposite and face the gap")
    gap = a.signed_distance_m(b.point_m)
    gap_rev = b.signed_distance_m(a.point_m)
    if (not isfinite(gap) or not isfinite(gap_rev) or gap <= 0 or gap_rev <= 0
            or not isclose(gap, gap_rev, rel_tol=1e-13, abs_tol=0.0)):
        raise UnsupportedProblem("plate normals must face an open positive-width gap")
    # Reject even zero or exactly cancelling charges authored inside metal.
    for charge in problem.charges:
        if not (a.signed_distance_m(charge.position_m) > 0 and
                b.signed_distance_m(charge.position_m) > 0):
            raise UnsupportedProblem("all point charges must lie strictly between the plates")
    sources = _sites(problem)
    if len(sources) > 32:
        raise UnsupportedProblem("parallel-plate image reference supports at most 32 active sites")
    return a, b, gap, sources


@dataclass(frozen=True, slots=True)
class ParallelPlateSolution:
    """Finite-order 3D image-reference solution; no FEMM data is substituted.

    Admitted local observations (``potential_at``, ``electric_field_at``,
    ``surface_charge_density_at``) are certified within 1 % and sign-correct
    against a derived finite-N remainder plus floating-point representational
    bound; samples outside the certified domain raise ``OutsideDomain``.
    Source-free uniform slabs and on-plate Dirichlet potentials are exact.
    """

    lower: ConductingPlane
    upper: ConductingPlane
    gap_m: float
    sources: tuple[PointCharge, ...]
    permittivity_f_per_m: float
    image_order: int
    conductor_results: tuple[ConductorResult, ...]

    @property
    def symmetry_axis(self) -> Vector3:
        """A natural meridian direction, not axial symmetry for general sources."""
        return self.lower.normal

    @property
    def _k(self) -> float:
        return 1.0 / (4.0*pi*self.permittivity_f_per_m)

    def _height(self, point: Vector3) -> float:
        return self.lower.signed_distance_m(point)

    def _slab_heights(self, point: Vector3) -> tuple[float, float]:
        """Signed distances to both plates with large-coordinate consistency check.

        The absolute floor 1e-14 is intentionally inert on sub-1e-14 tiny gaps,
        where extreme transverse tilt amplification is not the failure mode.
        """
        h_lo = self.lower.signed_distance_m(point)
        h_hi = self.upper.signed_distance_m(point)
        if (not (isfinite(h_lo) and isfinite(h_hi))
                or abs((h_lo + h_hi) - self.gap_m) > max(1e-12 * self.gap_m, 1e-14)):
            raise OutsideDomain("sample point exceeds parallel-plate geometric precision")
        return h_lo, h_hi

    def _background_v(self, height: float) -> float:
        v0 = _voltage(self.lower)
        return v0 + (_voltage(self.upper)-v0)*(height/self.gap_m)

    def _transverse(self, point: Vector3, source: PointCharge):
        delta = _minus(point, source.position_m)
        axial = _dot(delta, self.lower.normal)
        tangent = tuple(delta[i]-axial*self.lower.normal[i] for i in range(3))
        return tangent, hypot(*tangent)

    def _image_terms(self, height: float, source_height: float):
        """Reciprocal finite kernel, averaged over the two plate orientations.

        Positive images use n=-N..N. Negative images use the average of
        -N..N and 1-N..1+N. Averaging preserves exchange reciprocity AND
        plate-swap symmetry at finite order, removing the old midpoint bias.
        It does not make finite-order Dirichlet fields exact everywhere.
        """
        d = self.gap_m
        for n in range(-self.image_order, self.image_order+1):
            yield n, (height-source_height-2*n*d), (height+source_height-2*n*d), (height+source_height-2*(n+1)*d)

    def _source_potential_and_bound(self, point: Vector3, source: PointCharge, *,
                                    exclude_direct: bool = False) -> tuple[float, float]:
        _, radial = self._transverse(point, source)
        h = self._height(point)
        z0 = self._height(source.position_m)
        terms = []
        rep_terms = []
        for n, z1, z2, z3 in self._image_terms(h, z0):
            r2, r3 = hypot(radial, z2), hypot(radial, z3)
            if exclude_direct and n == 0:
                terms.append(-.5/r2-.5/r3)
                rep_terms.append(.5/r2+.5/r3)
                continue
            r1 = hypot(radial, z1)
            if r1 == 0:
                raise SingularField("potential is singular at an ideal point charge")
            # Rationalized reciprocal differences reduce distant-image cancellation.
            terms.append(.5*((z2-z1)/r2)*((z2+z1)/(r1+r2))/r1
                         + .5*((z3-z1)/r3)*((z3+z1)/(r1+r3))/r1)
            rep_terms.append(1.0/r1 + .5/r2 + .5/r3)
        value = self._k*source.charge_c*fsum(terms)
        d = self.gap_m
        u = min(max(h / d, 0.0), 1.0)
        u0 = z0 / d
        # Exact plate-swap symmetry: |R_N(u, u0)| == |R_N(1-u, 1-u0)|.
        b1 = _tail_bounds_orientation(d, self.image_order, u, u0, radial)[0]
        b2 = _tail_bounds_orientation(d, self.image_order, 1.0 - u, 1.0 - u0, radial)[0]
        bound = self._k * abs(source.charge_c) * (min(b1, b2) + 16.0 * _U_MACH * fsum(rep_terms))
        return value, bound

    def _source_potential(self, point: Vector3, source: PointCharge, *, exclude_direct=False):
        return self._source_potential_and_bound(point, source, exclude_direct=exclude_direct)[0]

    def _source_field_and_bounds(self, point: Vector3, source: PointCharge, *,
                                 exclude_direct: bool = False) -> tuple[Vector3, float, float]:
        tangent, radial = self._transverse(point, source)
        h = self._height(point)
        z0 = self._height(source.position_m)
        axial_terms, radial_terms = [], []
        rep_en_terms, rep_er_terms = [], []
        for n, z1, z2, z3 in self._image_terms(h, z0):
            if exclude_direct and n == 0:
                inv1 = 0.
            else:
                r1 = hypot(radial, z1)
                if r1 == 0:
                    raise SingularField("electric field is singular at an ideal point charge")
                inv1 = 1/r1**3
            inv2, inv3 = .5/hypot(radial, z2)**3, .5/hypot(radial, z3)**3
            axial_terms.append(z1*inv1-z2*inv2-z3*inv3)
            radial_terms.append(inv1-inv2-inv3)
            rep_en_terms.append(abs(z1)*inv1 + abs(z2)*inv2 + abs(z3)*inv3)
            rep_er_terms.append(radial*(inv1 + inv2 + inv3))
        axial, transverse = fsum(axial_terms), fsum(radial_terms)
        field = tuple(self._k*source.charge_c*(axial*self.lower.normal[i]+transverse*tangent[i])
                      for i in range(3))
        d = self.gap_m
        u = min(max(h / d, 0.0), 1.0)
        u0 = z0 / d
        # Exact plate-swap symmetry: |R_N(u, u0)| == |R_N(1-u, 1-u0)|.
        _, en1, er1 = _tail_bounds_orientation(d, self.image_order, u, u0, radial)
        _, en2, er2 = _tail_bounds_orientation(d, self.image_order, 1.0 - u, 1.0 - u0, radial)
        scale = self._k * abs(source.charge_c)
        b_en = scale * (min(en1, en2) + 16.0 * _U_MACH * fsum(rep_en_terms))
        b_er = scale * (min(er1, er2) + 16.0 * _U_MACH * fsum(rep_er_terms))
        return field, b_en, b_er

    def _source_field(self, point: Vector3, source: PointCharge, *, exclude_direct=False):
        return self._source_field_and_bounds(point, source, exclude_direct=exclude_direct)[0]

    def potential_at(self, point_m: Vector3) -> float:
        point = _position(point_m, "sample point")
        h_lo, h_hi = self._slab_heights(point)
        if h_lo <= 0:
            return _voltage(self.lower)
        if h_hi <= 0:
            return _voltage(self.upper)
        if not self.sources:
            return self._background_v(h_lo)
        parts = [self._source_potential_and_bound(point, q) for q in self.sources]
        value = fsum((self._background_v(h_lo), *(v for v, _ in parts)))
        bg_rep = 4.0 * _U_MACH * (abs(_voltage(self.lower)) + abs(_voltage(self.upper)))
        bound = fsum((bg_rep, *(b for _, b in parts)))
        _certify_total("potential", value, bound, self.image_order)
        return value

    def _field_inside_and_bounds(self, point: Vector3) -> tuple[Vector3, float, float]:
        """Vacuum-side E and certified (B_En, B_Erho) bounds, also evaluable on plates for sigma."""
        v_lo, v_hi = _voltage(self.lower), _voltage(self.upper)
        background = -(v_hi - v_lo) / self.gap_m
        if not self.sources:
            return tuple(background * self.lower.normal[i] for i in range(3)), 0.0, 0.0
        parts = [self._source_field_and_bounds(point, q) for q in self.sources]
        field = tuple(fsum((background*self.lower.normal[i], *(f[i] for f, _, _ in parts)))
                      for i in range(3))
        bg_rep_en = 4.0 * _U_MACH * (abs(v_lo) + abs(v_hi)) / self.gap_m
        b_en = fsum((bg_rep_en, *(ben for _, ben, _ in parts)))
        b_er = fsum(ber for _, _, ber in parts)
        return field, b_en, b_er

    def _field_inside(self, point: Vector3) -> Vector3:
        """Vacuum-side E, also evaluable on the plates for one-sided sigma."""
        return self._field_inside_and_bounds(point)[0]

    def force_energy(self):
        """Mechanical force/energy with only the direct point self-field omitted."""
        from .observables import _plate_observables
        return _plate_observables(self)

    def force_on_charge_n(self, index: int, *, include_direct: bool = True) -> Vector3:
        """Force on a grouped site; conductor contribution includes the bias field."""
        result = self.force_energy()
        total = result.force_on_charge_n(index)  # shared strict index validation
        return total if include_direct else result.charges[index].conductor_force_n

    @property
    def interaction_energy_j(self) -> float:
        """Mechanical potential, NOT total field energy of an infinite capacitor."""
        return self.force_energy().energy.total_j

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point = _position(point_m, "sample point")
        h_lo, h_hi = self._slab_heights(point)
        band = min(1e-14, 1e-12 * self.gap_m)
        if abs(h_lo) <= band or abs(h_hi) <= band:
            raise SingularField("electric field has distinct one-sided limits at a plate")
        if h_lo < 0 or h_hi < 0:
            return (0.0,0.0,0.0)
        field, b_en, b_er = self._field_inside_and_bounds(point)
        if self.sources:
            _certify_total("electric_field", hypot(*field), hypot(b_er, b_en), self.image_order)
        return field

    def surface_charge_density_at(self, surface: str, point_m: Vector3,
                                  *, conductor_name: str | None = None) -> float:
        """Local sigma on the vacuum-facing side of the named infinite plate."""
        if surface != "outer":
            raise ValueError("each ideal plate exposes only one vacuum-facing surface")
        if conductor_name not in (self.lower.name,self.upper.name):
            raise ValueError("conductor_name must name one of the two plates")
        plate = self.lower if conductor_name == self.lower.name else self.upper
        point = _position(point_m, "surface point")
        tol = min(1e-11, 1e-9 * self.gap_m)
        if abs(plate.signed_distance_m(point)) > tol:
            raise ValueError("surface-density point must lie on the requested plate")
        self._slab_heights(point)
        field, b_en, _ = self._field_inside_and_bounds(point)
        sigma = self.permittivity_f_per_m*_dot(field,plate.normal)
        if self.sources:
            _certify_total("surface_charge_density", sigma,
                           self.permittivity_f_per_m * b_en, self.image_order)
        return sigma


class ParallelPlateBackend:
    """Two facing infinite plates; analytical field uses truncated images.

    image_order=N averages the two plate-oriented truncations at n=-N..N.
    The reciprocal kernel preserves plate-swap symmetry at every order. Increasing N improves convergence but costs O(sources*N)
    work per field sample. Local V/E/sigma queries are certified within 1 % and
    sign-correct (raising OutsideDomain otherwise). The source-free uniform field is exact.
    """

    def __init__(self, *, image_order: int = 96):
        if isinstance(image_order,bool) or not isinstance(image_order,int) or not 1 <= image_order <= 512:
            raise ValueError("image_order must be an integer in [1,512]")
        self.image_order = image_order

    def supports(self, problem: ElectrostaticProblem) -> bool:
        try:
            _layout(problem)
        except UnsupportedProblem:
            return False
        return True

    def solve(self, problem: ElectrostaticProblem) -> ParallelPlateSolution:
        a,b,d,sources = _layout(problem)
        v0,v1 = _voltage(a),_voltage(b)
        if v0 == v1:
            # Finite source-induced *total* charge on either infinite face,
            # via the exact weighting potentials z/d and 1-z/d.
            qa = -fsum(q.charge_c*(1-a.signed_distance_m(q.position_m)/d)
                       for q in sources)
            qb = -fsum(q.charge_c*(a.signed_distance_m(q.position_m)/d)
                       for q in sources)
            results = (
                ConductorResult(a.name,v0,qa,0.0,qa),
                ConductorResult(b.name,v1,qb,0.0,qb),
            )
        else:
            # Different Dirichlet values create an infinite uniform capacitor
            # surface density; its integral is unbounded on infinite plates.
            results = (
                ConductorResult(a.name,v0,None,0.0,None),
                ConductorResult(b.name,v1,None,0.0,None),
            )
        return ParallelPlateSolution(a,b,d,sources,problem.permittivity_f_per_m,
                                     self.image_order,results)
