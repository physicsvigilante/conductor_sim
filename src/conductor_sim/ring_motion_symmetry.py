"""M27b: prove one ideal full-ring source preserves an actual motion plane.

This is a physical symmetry certificate, not a display projection or a
numerical tolerance used to force an otherwise 3D orbit onto a chosen plane.
Each ring must itself be reflection-invariant. Collective paired-ring
reflection is a separate unimplemented physical regime.
"""
from __future__ import annotations

from math import hypot, ulp

from .prescribed_fields import UniformRingField, _dot, _mul, _sub, _unit
from .solver import UnsupportedProblem


def certify_ring_motion_plane(source: UniformRingField, *, origin_m, normal) -> str:
    """Return 'meridian' or 'equatorial'; refuse any other physical plane.

    Meridian: plane through ring centre and containing its symmetry axis.
    Equatorial: ring's own plane through its centre, normal parallel to axis.
    Tolerance is derived from the actual ring geometry/roundoff, not an
    authored numerical length_scale that could legalize physical asymmetry.
    """
    if type(source) is not UniformRingField:
        raise TypeError("M27b ring-plane certificate requires UniformRingField")
    unit_normal = _unit(normal, "physical motion-plane normal")
    separation = abs(_dot(_sub(source.center_m, origin_m), unit_normal))
    world_scale = max(source.radius_m, *(abs(x) for x in
                       (*source.center_m, *origin_m)))
    tolerance = max(1e-11*source.radius_m, 64*ulp(world_scale))
    if separation > tolerance:
        raise UnsupportedProblem(
            "full-ring centre does not lie in the physical motion plane")
    cosine = _dot(source.axis, unit_normal)
    if abs(cosine) <= 1e-11:
        return "meridian"
    axial_parallel_error = hypot(*_sub(source.axis, _mul(unit_normal, cosine)))
    if axial_parallel_error <= 1e-11:
        return "equatorial"
    raise UnsupportedProblem(
        "full-ring axis is oblique: plane is neither meridian nor equatorial")
