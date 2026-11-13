"""Independent physical keys and test-only trajectory reference (standard library).

No conductor_sim, authored-submission or production RK4 imports. Nonlinear
reference paths use fine velocity-Verlet and a separate half-step convergence
check, not the production RK4 method. The elementary-electron plate case uses
exact uniform-field kinematics with an explicit negligible-induction bound.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, fsum, hypot, pi

EPS0=8.8541878128e-12  # fixed permittivity used by this benchmark, in F/m
K=1/(4*pi*EPS0)

@dataclass(frozen=True)
class Key:
    backend: str | None = None
    charge: float = 0.
    mass: float = 1e-6
    position: tuple = (0.,0.,0.)
    velocity: tuple = (0.,0.,0.)
    duration: float = .1
    interval: float = .005
    center: tuple = (0.,0.,0.)
    normal: tuple = (0.,0.,1.)
    # Spheres: (a,b,Q), b=0 for solid; Q=None denotes grounding.
    shells: tuple = ()
    gap: float = 0.
    voltages: tuple = (0.,0.)
    refusal_words: tuple = ()

KEYS = {
    'tilted-plane':Key('plane',-.8e-9,position=(.43,-.128,.396),velocity=(.13,.009,.012),
        duration=.16,interval=.008,center=(.4,-.2,.3),normal=(0.,.6,.8)),
    'electron-plates':Key('plates',-1.602176634e-19,mass=9.1093837139e-31,
        position=(0.,0.,.1),velocity=(8e5,2e5,-1e5),duration=5e-8,interval=2.5e-9,
        gap=.2,voltages=(0.,20.)),
    'grounded-sphere':Key('sphere',-1e-9,position=(.38,-.26,.1),velocity=(-.02,.18,.06),
        duration=.12,interval=.006,center=(.2,-.3,.1),shells=((.07,0.,None),)),
    'isolated-sphere':Key('sphere',.7e-9,position=(.15,0.,0.),velocity=(-.04,.11,0.),
        duration=.08,interval=.004,shells=((.06,0.,1.6e-9),)),
    'cavity-oblique':Key('cavity',-1.1e-9,position=(.025,-.01,.005),velocity=(.02,.11,-.04),
        duration=.1,interval=.005,shells=((.12,.15,None),)),
    'nested-cavity':Key('nested',.6e-9,position=(-.48,.26,.07),velocity=(.06,-.04,.03),
        duration=.08,interval=.004,center=(-.5,.25,.1),shells=((.12,.15,-.2e-9),(.24,.28,1e-9))),
    'rotated-radial':Key('sphere',1e-9,position=(.1,.2,.3),velocity=(-.01,-.02,-.03),
        duration=.06,interval=.003,shells=((.01,0.,None),)),
    'centre-crossing':Key('cavity',.2e-9,position=(.02,0.,0.),velocity=(-.25,0.,0.),
        duration=.12,interval=.006,shells=((.12,.16,.3e-9),)),
    'two-moving':Key(refusal_words=('one','moving','coupled')),
    'numerical-femm':Key(refusal_words=('femm','numerical','force')),
    'finite-plate-edge':Key(refusal_words=('infinite','finite','fringing')),
    'varying-voltage':Key(refusal_words=('fixed','voltage','time-dependent')),
}

def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return fsum(x*y for x,y in zip(a,b))
def norm(v):return hypot(*v)
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def force_energy(key,position):
    """Kelvin/mirror closed forms, differentiated independently, in SI.

    Cavity outer-shell constants come from physical surface-charge potentials:
    each inner face has -enclosed charge; each outer face has enclosed+Q.
    This avoids calling an engine solution or reusing its image construction.
    """
    q=key.charge;rvec=sub(position,key.center)
    if key.backend=='plane':
        h=dot(rvec,key.normal)
        return tuple(-K*q*q*n/(4*h*h) for n in key.normal),-K*q*q/(4*h)
    if key.backend=='plates':
        E=-(key.voltages[1]-key.voltages[0])/key.gap
        h=dot(rvec,key.normal)
        return tuple(q*E*n for n in key.normal),q*(key.voltages[0]-E*h)
    radius=norm(rvec);a,_,Q=key.shells[0]
    if key.backend=='sphere':
        multiplier=-K*q*q*a/(radius*radius-a*a)**2
        energy=-K*q*q*a/(2*(radius*radius-a*a))
        if Q is not None:
            multiplier+=K*q*Q/radius**3+K*q*q*a/radius**4
            energy+=K*q*Q/radius+K*q*q*a/(2*radius*radius)
        return tuple(multiplier*x for x in rvec),energy
    multiplier=K*q*q*a/(a*a-radius*radius)**2
    energy=-K*q*q*a/(2*(a*a-radius*radius))
    # These trial keys either have one grounded shell or only fixed-Q shells.
    if Q is not None:
        def cavity_voltage(source, include_shell_charge):
            enclosed=source;value=0.
            for i,(inner,outer,net) in enumerate(key.shells):
                if i: value-=K*enclosed/inner
                enclosed+=net if include_shell_charge else 0.
                value+=K*enclosed/outer
            return value
        energy+=.5*q*cavity_voltage(q,False)+q*cavity_voltage(0.,True)
    return tuple(multiplier*x for x in rvec),energy


def times(key):
    return [0.]+[min(key.duration,i*key.interval)
        for i in range(1,ceil(key.duration/key.interval)+1)]


def _row(key,t,p,v):
    force,U=force_energy(key,p)
    kinetic=.5*key.mass*norm(v)**2
    return dict(time_s=t,position_m=p,velocity_m_s=v,force_n=force,
                acceleration_m_s2=tuple(x/key.mass for x in force),
                kinetic_energy_j=kinetic,interaction_energy_j=U,total_energy_j=kinetic+U)


@lru_cache(maxsize=32)
def trajectory(case_id, substeps=256):
    """Reference calculation only; never used to drive a production viewer."""
    key=KEYS[case_id]
    if not key.backend:raise ValueError('unsupported case has no trajectory reference')
    if key.backend=='plates':
        acc=tuple(x/key.mass for x in force_energy(key,key.position)[0])
        rows=tuple(_row(key,t,
            tuple(key.position[i]+key.velocity[i]*t+.5*acc[i]*t*t for i in range(3)),
            tuple(key.velocity[i]+acc[i]*t for i in range(3))) for t in times(key))
        # For this modest-gap elementary-charge trajectory the deliberately
        # conservative image-force/energy bounds are tiny, but not zero.
        d=min(min(r['position_m'][2],key.gap-r['position_m'][2]) for r in rows)
        assert d>.05
        assert K*key.charge**2/d**2 < 1e-8*norm(rows[0]['force_n'])
        return rows
    p,v=key.position,key.velocity
    a=tuple(f/key.mass for f in force_energy(key,p)[0])
    rows=[_row(key,0.,p,v)];old=0.
    for t in times(key)[1:]:
        dt=(t-old)/substeps
        for _ in range(substeps):
            pn=tuple(p[i]+v[i]*dt+.5*a[i]*dt*dt for i in range(3))
            an=tuple(f/key.mass for f in force_energy(key,pn)[0])
            vn=tuple(v[i]+.5*(a[i]+an[i])*dt for i in range(3))
            p,v,a=pn,vn,an
        rows.append(_row(key,t,p,v));old=t
    return tuple(rows)


def convergence(case_id):
    coarse,fine=trajectory(case_id,128),trajectory(case_id,256)
    return dict(position_difference_m=max(norm(sub(a['position_m'],b['position_m'])) for a,b in zip(coarse,fine)),
                velocity_difference_m_s=max(norm(sub(a['velocity_m_s'],b['velocity_m_s'])) for a,b in zip(coarse,fine)))
