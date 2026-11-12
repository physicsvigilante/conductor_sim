"""Dimensionless Kelvin image term, avoiding a far-away overflowing image.

For x and y relative to the sphere center, set d=|y|, u=y/d,
w=(d/a)*(x/a)-u. The image potential divided by k*q is
-1/(a*|w|); its electric field divided by k*q is
-(d/a)*w/(a**2*|w|**3). This is exactly the ordinary image
construction rearranged; at d=0 its continuous limit is (-1/a, 0).
"""
from math import hypot
from .model import Vector3


def image_term(point: Vector3, source: Vector3, center: Vector3,
               radius: float) -> tuple[float, Vector3]:
    y = tuple(source[i]-center[i] for i in range(3))
    d = hypot(*y)
    if d == 0:
        return -1/radius, (0., 0., 0.)
    t = d/radius
    w = tuple(t*((point[i]-center[i])/radius)-y[i]/d for i in range(3))
    length = hypot(*w)
    factor = -t/(radius*radius*length**3)
    return -1/(radius*length), tuple(factor*v for v in w)
