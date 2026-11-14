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


@dataclass(frozen=True, slots=True)
class RodReflectionCertificate:
    """Reflection closure of whole, unoriented finite uniform-charge rods.

    ``pairs`` references the input's unchanged ordered inventory. ``invariant_indices``
    have *whole segments* invariant under reflection (not merely their centres).
    Equal signed total charge and equal reflected end points imply equal
    line-density measure. Only floating-point representation is tolerated.
    """
    origin_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    invariant_indices: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    tolerance_m: float
    max_pair_residual_m: float


def certify_rod_reflection(sources, *, origin_m, normal):
    """Certify reflection closure of 1..64 *whole* fixed line-charge segments.

    Endpoint order is not physical. Each off-plane non-invariant rod must
    have exactly one partner with equal *signed* total charge and reflected
    complete segment, preserving multiplicity. Unlike a pointwise force test,
    this proves physical-plane invariance throughout the qualified domain.
    A bounded deterministic search handles crossing rods whose midpoints are
    on the plane but whose individually tilted geometries are not invariant.
    """
    from .prescribed_fields import UniformRodField
    if type(sources) not in (list, tuple) or not 1 <= len(sources) <= 64:
        raise ValueError('rod reflection requires 1 to 64 explicit sources')
    sources = tuple(sources)
    if any(type(s) is not UniformRodField for s in sources):
        raise TypeError('rod reflection requires built-in UniformRodField only')
    origin = _vector(origin_m, 'rod reflection origin')
    n = _unit(normal, 'rod reflection normal')
    ends = tuple((_sub(s.start_m, origin), _sub(s.end_m, origin)) for s in sources)
    span = max(hypot(*p) for pair in ends for p in pair)
    span = max(span, *(hypot(*_sub(pair[1],pair[0])) for pair in ends))
    if not isfinite(span) or not span:
        raise UnsupportedProblem('rod reflection geometry has no finite local span')
    tolerance = 64*max(ulp(span), *(ulp(v) for v in origin),
                       *(ulp(v) for rod in sources for p in (rod.start_m,rod.end_m) for v in p))
    if tolerance > 1e-11*span:
        raise UnsupportedProblem('rod reflection geometry is poorly resolved; translate/scale the authored coordinates')
    def reflect(p):
        return _sub(p, _mul(n, 2*_dot(p, n)))
    mirrored = tuple((reflect(a), reflect(b)) for a,b in ends)
    def distance(one, two):
        a,b=one; c,d=two
        return min(max(hypot(*_sub(a,c)),hypot(*_sub(b,d))),
                   max(hypot(*_sub(a,d)),hypot(*_sub(b,c))))
    def residual(i,j):
        return max(distance(mirrored[i], ends[j]),distance(mirrored[j],ends[i]))
    invariant = tuple(i for i in range(len(sources)) if residual(i,i) <= tolerance)
    unmatched = set(range(len(sources)))-set(invariant)
    adjacency = {i: tuple(j for j in sorted(unmatched) if j!=i and
                           sources[i].charge_c == sources[j].charge_c and
                           residual(i,j) <= tolerance) for i in unmatched}
    def fail():
        raise UnsupportedProblem('rod inventory is not reflection symmetric: every full segment requires a distinct equal-signed-charge reflected partner')
    if len(unmatched)%2 or any(not adjacency[i] for i in unmatched):
        fail()
    pairs=[]
    visited=0
    def match():
        nonlocal visited
        if not unmatched:
            return True
        visited+=1
        # Explicit guard: approximate near-coincident geometry must not turn
        # physical qualification into an unbounded combinatorial search.
        if visited>50000:
            raise UnsupportedProblem('rod reflection matching is ambiguous/poorly resolved')
        i=min(unmatched, key=lambda k:(sum(j in unmatched for j in adjacency[k]),
                                      sources[k].charge_c,sources[k].start_m,
                                      sources[k].end_m,k))
        choices=[j for j in adjacency[i] if j in unmatched]
        unmatched.remove(i)
        for j in choices:
            unmatched.remove(j)
            pairs.append((min(i,j),max(i,j)))
            if match():return True
            pairs.pop();unmatched.add(j)
        unmatched.add(i)
        return False
    if not match():
        fail()
    maximum=max((residual(i,j) for i,j in pairs),default=0.)
    return RodReflectionCertificate(origin,n,invariant,tuple(sorted(pairs)),
                                    tolerance,maximum)
