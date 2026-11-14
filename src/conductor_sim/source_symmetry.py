"""Bounded geometric reflection certificates for fixed point-source inventories.

A certificate concerns the *whole signed inventory*, not force cancellation at
one probe. It never moves/merges sources, repairs geometry, changes a field law
or constrains a trajectory. Other source families retain their own contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, ulp

from .prescribed_fields import FixedPointField, _dot, _mul, _sub, _unit, _vector
from .solver import UnsupportedProblem


@dataclass(frozen=True, slots=True)
class PointReflectionCertificate:
    """Indices refer to the original ordered point inventory supplied to certify.

    ``pairs`` is a one-to-one involution. Each singleton lies in the plane.
    Tolerance accounts only for floating-point representation and reflection
    arithmetic, not an approximate physical symmetry or source charge rounding.
    """
    origin_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    in_plane_indices: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    tolerance_m: float
    max_pair_residual_m: float


def certify_point_reflection(sources, *, origin_m, normal):
    """Prove reflection closure of 1..64 built-in fixed point sources.

    Equal *signed* charge is required exactly. Every off-plane source must
    match one source on the opposite side at its reflected position, with
    full multiplicity. A deterministic bipartite matching avoids a greedy
    order-dependent result for repeated or roundoff-close source positions.

    The geometric tolerance is 64 ulps of the local span/world coordinates.
    Poorly resolved coordinates (tolerance > 1e-11 of the local span) are
    refused; a huge world translation must not hide real asymmetry. The
    caller's numerical ODE tolerance and length scale cannot relax this proof.
    """
    if type(sources) not in (tuple, list) or not 1 <= len(sources) <= 64:
        raise ValueError('reflection requires 1 to 64 explicit point sources')
    sources = tuple(sources)
    if any(type(s) is not FixedPointField for s in sources):
        raise TypeError('reflection certificate accepts built-in FixedPointField only')
    origin = _vector(origin_m, 'reflection origin')
    n = _unit(normal, 'reflection normal')
    delta = tuple(_sub(s.position_m, origin) for s in sources)
    span = max(hypot(*d) for d in delta)
    if not isfinite(span) or span == 0:
        raise UnsupportedProblem('reflection geometry has no finite resolved local span')
    tolerance = 64 * max(ulp(span), *(ulp(x) for x in origin),
                         *(ulp(x) for s in sources for x in s.position_m))
    if tolerance > 1e-11 * span:
        raise UnsupportedProblem('reflection geometry is poorly resolved; translate/scale authored coordinates')
    height = tuple(_dot(d, n) for d in delta)
    on_plane = tuple(i for i, h in enumerate(height) if abs(h) <= tolerance)
    positive = [i for i, h in enumerate(height) if h > tolerance]
    negative = [i for i, h in enumerate(height) if h < -tolerance]

    def fail():
        raise UnsupportedProblem('fixed point-source inventory is not reflection symmetric: '
                                 'equal signed charges and mirrored positions are required')

    if len(positive) != len(negative):
        fail()
    # Reflection is checked in both directions; no position is rounded/snap-fit.
    reflected = tuple(_sub(d, _mul(n, 2*h)) for d, h in zip(delta, height))
    residual = {(i, j): max(hypot(*_sub(delta[j], reflected[i])),
                            hypot(*_sub(delta[i], reflected[j])))
                for i in positive for j in negative}
    key = lambda i: (sources[i].charge_c, sources[i].position_m, i)
    candidates = {i: sorted((j for j in negative
                            if sources[i].charge_c == sources[j].charge_c
                            and residual[i, j] <= tolerance), key=key)
                  for i in positive}
    matched = {}

    def assign(i, visited):
        for j in candidates[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in matched or assign(matched[j], visited):
                matched[j] = i
                return True
        return False

    for i in sorted(positive, key=key):
        if not assign(i, set()):
            fail()
    pairs = tuple(sorted((min(i, j), max(i, j)) for j, i in matched.items()))
    maximum = max((residual[i, j] for j, i in matched.items()), default=0.)
    return PointReflectionCertificate(origin, n, on_plane, pairs, tolerance, maximum)
