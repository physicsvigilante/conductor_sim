"""Independent M13c physical keys and elementary image-charge reference.

No imports of the authored submission, engine solvers, force_energy, or sweep
sampler. All expected geometry is repeated here from the *frozen prompts*.
For parallel plates an independently accumulated symmetric image series is
used at greater order than the candidate solver; it is still approximate.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import fsum, hypot, pi

EPS = 8.8541878128e-12
K = 1.0/(4*pi*EPS)
# Keys are plain SI data, never an engine ElectrostaticProblem or solver result.


def p(point=(0.,0.,0.), normal=(0.,0.,1.), name='plane', volts=0.):
    return ('plane', point, normal, 'voltage' if volts else 'grounded', volts, name)


def s(radius, name='sphere', *, fixed=None):
    return ('sphere', radius, (0.,0.,0.), 'charge' if fixed is not None else 'grounded',
            0. if fixed is None else fixed, name)


def plates(gap=.2, low=0., high=0.):
    return (p(name='lower',volts=low),
            p((0.,0.,gap),(0.,0.,-1.),name='upper',volts=high))


@dataclass(frozen=True, slots=True)
class Key:
    backend: str | None = None
    conductors: tuple = ()
    charges: tuple = ()  # (charge_C, world_position_m)
    selected: int = 0
    end: tuple = ()
    steps: int = 0
    # Unsupported reasons must mention discriminating physical concepts.
    refusal_terms: tuple = ()


def supported(backend, conductors, charges, end, steps):
    return Key(backend, tuple(conductors), tuple(charges), 0, end, steps)


def blocked(*words):
    return Key(refusal_terms=tuple(words))


KEYS = {
    'plane-approach': supported('plane',[p(name='earth')],
        [(1e-9,(0.,0.,.18))],(0.,0.,.06),7),
    'plane-two-signed': supported('plane',[p(name='ground')],
        [(.8e-9,(0.,0.,.06)),(-.35e-9,(.045,0.,.08))],(0.,0.,.13),7),
    'plane-rotated': supported('plane',[p((.20,0.,0.),(1.,0.,0.),'side')],
        [(-.6e-9,(.30,.02,0.))],(.40,.02,0.),5),
    'sphere-grounded': supported('sphere',[s(.11)],
        [(1e-9,(0.,0.,.17))],(0.,0.,.30),7),
    'sphere-isolated': supported('sphere',[s(.12,fixed=1.8e-9)],
        [(.7e-9,(0.,0.,.22))],(0.,0.,.35),7),
    'sphere-two-sources': supported('sphere',[s(.10)],
        [(.8e-9,(0.,0.,.19)),(-.2e-9,(.24,.10,.03))],(0.,0.,.30),7),
    'plates-grounded': supported('plates',plates(),
        [(1e-9,(0.,0.,.05))],(0.,0.,.15),9),
    'plates-biased-negative': supported('plates',plates(.24,0.,20.),
        [(-.8e-9,(0.,0.,.06))],(0.,0.,.18),7),
    'plates-oblique-mixed': supported('plates',plates(),
        [(.65e-9,(-.04,0.,.05)),(-.23e-9,(-.02,.06,.11))],(.045,0.,.14),7),
    'plates-dual-bias': supported('plates',plates(.2,-4.,8.),
        [(.9e-9,(0.,0.,.06))],(0.,0.,.14),5),
    'dipole-torque': blocked('dipole','torque','rotat'),
    'ring-field': blocked('ring','sphere'),
    'insulating-volume': blocked('insulat','conduct'),
    'voltage-change-during-motion': blocked('volt','chang','work'),
    'connected-spheres': blocked('connect','sphere','constraint'),
    'time-trajectory': blocked('time','velocity','trajectory'),
    'offaxis-femm-force': blocked('femm','off-axis','force'),
    'charge-in-cavity-motion': blocked('cavity','force'),
}


def add(a,b):
    return tuple(a[i]+b[i] for i in range(3))


def sub(a,b):
    return tuple(a[i]-b[i] for i in range(3))


def mul(v,k):
    return tuple(k*x for x in v)


def dot(a,b):
    return fsum(a[i]*b[i] for i in range(3))


def unit(v):
    return mul(v,1/hypot(*v))


def source_positions(key, distance):
    origin=key.charges[key.selected][1]
    path=sub(key.end,origin)
    length=hypot(*path)
    locations=[(q,tuple(r)) for q,r in key.charges]
    locations[key.selected]=(locations[key.selected][0],add(origin,mul(path,distance/length)))
    return locations


def images(key, q, r, *, cutoff=768):
    if key.backend=='plane':
        surface,n=key.conductors[0][1:3]
        height=dot(sub(r,surface),n)
        return [(-q,sub(r,mul(n,2*height)))]
    if key.backend=='sphere':
        sphere=key.conductors[0]
        radius,center=sphere[1:3]
        delta=sub(r,center)
        length=hypot(*delta)
        mirror=add(center,mul(delta,(radius/length)**2))
        images=[(-q*radius/length,mirror)]
        if sphere[3]=='charge':
            images.append((q*radius/length,center))
        return images
    low,high=key.conductors
    origin,n=low[1:3]
    gap=dot(sub(high[1],origin),n)
    height=dot(sub(r,origin),n)
    # Real source n=0 belongs to the direct Coulomb term, excluded below.
    sources=[]
    for i in range(-cutoff,cutoff+1):
        if i:
            sources.append((q,add(r,mul(n,2*i*gap))))
        reflected=sub(r,mul(n,2*height))
        sources.append((-.5*q,add(reflected,mul(n,2*i*gap))))
        sources.append((-.5*q,add(reflected,mul(n,2*(i+1)*gap))))
    return sources


def reference_observation(key: Key, distance: float):
    """Return (total force vector, U_direct, U_induced, U_background) in SI.

    No engine field/force/energy evaluation is used. Finite plate cutoff is
    768, compared with candidate backend cutoff 96. The test also checks a
    closed-form midpoint slab result and plane energy to avoid common bias.
    """
    sources=source_positions(key,distance)
    charges=[q for q,_ in sources]
    positions=[r for _,r in sources]
    selected=key.selected
    forces=[[],[],[]]
    direct_energy=[]
    for i,(qi,ri) in enumerate(sources):
        for j in range(i+1,len(sources)):
            qj,rj=sources[j]
            delta=sub(ri,rj)
            d=hypot(*delta)
            value=K*qi*qj/d
            direct_energy.append(value)
            f=mul(delta,value/d**2)
            if i==selected:
                for k in range(3):forces[k].append(f[k])
            if j==selected:
                for k in range(3):forces[k].append(-f[k])
    induced_energy=[]
    for i,(qi,ri) in enumerate(sources):
        for qj,rj in sources:
            for qimg,rimg in images(key,qj,rj):
                diff=sub(ri,rimg)
                d=hypot(*diff)
                value=K*qi*qimg/d
                induced_energy.append(value*.5)
                if i==selected:
                    fi=mul(diff,value/d**2)
                    for k in range(3):forces[k].append(fi[k])
    bg_energy=[]
    if key.backend=='sphere' and key.conductors[0][3]=='charge':
        key_sphere=key.conductors[0]
        center=key_sphere[2]
        qfixed=key_sphere[4]
        for i,(qi,ri) in enumerate(sources):
            delta=sub(ri,center)
            d=hypot(*delta)
            value=K*qi*qfixed/d
            bg_energy.append(value)
            if i==selected:
                fi=mul(delta,value/d**2)
                for k in range(3):forces[k].append(fi[k])
    elif key.backend=='plates':
        low,high=key.conductors
        vlow,vhigh=low[4],high[4]
        n=low[2]
        gap=dot(sub(high[1],low[1]),n)
        for i,(qi,ri) in enumerate(sources):
            h=dot(sub(ri,low[1]),n)
            bg_energy.append(qi*(vlow+(vhigh-vlow)*h/gap))
            if i==selected:
                fi=mul(n,-qi*(vhigh-vlow)/gap)
                for k in range(3):forces[k].append(fi[k])
    return (tuple(fsum(f) for f in forces), fsum(direct_energy),
            fsum(induced_energy), fsum(bg_energy))
