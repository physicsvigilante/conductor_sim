"""Independent SI grading keys and boundary Green functions (standard library).

No engine, submission, production image builder or RK4 imports. The symmetric
sphere kernel avoids image locations entirely, including the cavity centre.
Nonlinear reference paths use test-only velocity-Verlet with step halving.
These calculations must never supply data to the production viewer.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import fsum, hypot, pi, sqrt

EPS0=8.8541878128e-12
K=1/(4*pi*EPS0)

@dataclass(frozen=True)
class Key:
    backend: str | None = None
    sources: tuple = ()  # (charge in C, position in m), original author order
    moving: int = 0
    mass: float = 1e-6
    velocity: tuple = (0.,0.,0.)
    duration: float = .1
    center: tuple = (0.,0.,0.)
    normal: tuple = (0.,0.,1.)
    # (inner radius, outer radius or zero for a solid, Q or None for ground)
    shells: tuple = ()
    gap: float = .2
    voltages: tuple = (0.,0.)
    refusal_words: tuple = ()

KEYS={
 'plane-three':Key('plane',((.4e-9,(-.055,0,.14)),(-.7e-9,(.01,0,.10)),(-.2e-9,(.075,0,.16))),
     1,velocity=(.11,0,-.015),duration=.08),
 'tilted-plane':Key('plane',((-.3e-9,(.35,-.11,.42)),(.5e-9,(.42,-.14,.38))),
     1,mass=2e-6,velocity=(.09,-.006,-.008),duration=.10,center=(.4,-.2,.3),normal=(0,.6,.8)),
 'grounded-sphere':Key('sphere',((.6e-9,(.37,-.1,.3)),(-.3e-9,(.43,-.02,.3))),
     velocity=(.015,.12,0),duration=.08,center=(.2,-.1,.3),shells=((.06,0.,None),)),
 'isolated-sphere':Key('sphere',((.3e-9,(.22,-.05,0)),(-.5e-9,(.16,0,0)),(-.2e-9,(-.19,.07,0))),
     1,velocity=(.02,.10,0),duration=.06,shells=((.055,0.,1.2e-9),)),
 'cavity-three':Key('cavity',((.2e-9,(-.045,.02,0)),(-.15e-9,(.04,-.03,0)),(.4e-9,(.01,.035,0))),
     2,mass=1.5e-6,velocity=(.08,-.02,0),duration=.08,shells=((.12,.15,None),)),
 'nested-cavity':Key('nested',((-.2e-9,(-.34,.22,.1)),(.5e-9,(-.28,.21,.1))),
     1,velocity=(.04,.07,0),duration=.10,center=(-.3,.2,.1),
     shells=((.12,.15,.4e-9),(.23,.27,None))),
 'biased-plates':Key('plates',((.08e-9,(-.04,0,.13)),(-.12e-9,(.01,0,.08))),
     1,velocity=(.11,0,.02),duration=.12,voltages=(3.,23.)),
 'centre-release':Key('cavity',((.06e-9,(0,.04,0)),(-.05e-9,(0,0,.05)),(.10e-9,(0,0,0))),
     2,duration=.20,shells=((.12,.16,None),)),
 'two-moving':Key(refusal_words=('one','moving','coupled')),
 'nonplanar':Key(refusal_words=('invariant','plane','3d')),
 'numerical-femm':Key(refusal_words=('femm','numerical','force')),
 'varying-voltage':Key(refusal_words=('fixed','voltage','time-dependent')),
}

def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def norm(a):return hypot(*a)
def dot(a,b):return fsum(x*y for x,y in zip(a,b))


def _kernel(key,r,s,order):
    """Induced potential kernel H and minus-gradient w.r.t. its FIRST point.

    Sphere H=-a/sqrt(r²s²-2a² r.s+a⁴). It is symmetric in r,s, so
    differentiating 1/2 sum(q_i q_j H_ij) yields the physical image force.
    The direct singular Coulomb term is handled separately.
    """
    if key.backend=='plane':
        delta=sub(r,tuple(s[i]-2*dot(s,key.normal)*key.normal[i] for i in range(3)))
        d=norm(delta)
        return -1/d,tuple(-x/d**3 for x in delta)
    if key.backend=='plates':
        # Independently pair the ordinary lower-plane image construction at
        # n=-N..N. Unlike the production two-orientation averaging, this is
        # evaluated at high order and convergence is explicitly checked.
        hr,hs=dot(r,key.normal),dot(s,key.normal)
        tangent=tuple(r[i]-s[i]-(hr-hs)*key.normal[i] for i in range(3))
        rho2=dot(tangent,tangent);pot=[];axial=[];transverse=[]
        for n in range(-order,order+1):
            zp=hr-hs-2*n*key.gap
            zn=hr+hs-2*n*key.gap
            invn=1/sqrt(rho2+zn*zn);fn=invn**3
            if n==0:
                invp=fp=0.  # direct real-site term is counted separately
            else:
                invp=1/sqrt(rho2+zp*zp);fp=invp**3
            pot.append(invp-invn)
            axial.append(zp*fp-zn*fn)
            transverse.append(fp-fn)
        ax,tr=fsum(axial),fsum(transverse)
        return fsum(pot),tuple(ax*key.normal[i]+tr*tangent[i] for i in range(3))
    a=key.shells[0][0];ss=dot(s,s)
    D=sqrt(dot(r,r)*ss-2*a*a*dot(r,s)+a**4)
    return -a/D,tuple(-a*(r[i]*ss-a*a*s[i])/D**3 for i in range(3))


def _surface_voltage(key,source_charge,background):
    """Independent Gauss-law linear solve for shell outer-surface charges.

    Each inner face has minus the enclosed charge. Unknowns T1..Tn are
    the outer-face charges. Fixed-Q equations: T(i+1)-Ti=Qi. Grounded
    equations: sum of the physical spherical-surface potentials is zero.
    This does not import the production radial-voltage summation.
    """
    shells=key.shells;n=len(shells);matrix=[]
    for i,(_,b,Q) in enumerate(shells):
        row=[0.]*n
        if Q is not None:
            row[i]=1.
            if i:row[i-1]=-1.
            rhs=(Q if background else 0.)+(source_charge if i==0 else 0.)
        else:
            row[i]+=1/b
            for j in range(i+1,n):
                row[j-1]-=1/shells[j][0];row[j]+=1/shells[j][1]
            rhs=0.
        matrix.append(row+[rhs])
    for col in range(n):
        pivot=max(range(col,n),key=lambda i:abs(matrix[i][col]))
        matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
        scale=matrix[col][col]
        if scale==0:raise ValueError('singular reference shell constraints')
        matrix[col]=[v/scale for v in matrix[col]]
        for i in range(n):
            if i!=col:
                factor=matrix[i][col]
                matrix[i]=[a-factor*b for a,b in zip(matrix[i],matrix[col])]
    outer=[r[-1] for r in matrix]
    return K*fsum([outer[0]/shells[0][1]]+
        [-outer[j-1]/shells[j][0]+outer[j]/shells[j][1] for j in range(1,n)])


def force_energy(key,position,*,order=256,energy=True):
    """Force on moving particle and the full-inventory mechanical potential."""
    sources=list(key.sources);q=sources[key.moving][0];sources[key.moving]=(q,position)
    local=[(c,sub(p,key.center)) for c,p in sources];r=local[key.moving][1]
    forces=[[],[],[]];direct=[];induced=[];background=0.
    for j,(charge,s) in enumerate(local):
        if j!=key.moving:
            dvec=sub(r,s);d=norm(dvec)
            for i in range(3):forces[i].append(K*q*charge*dvec[i]/d**3)
        _,field=_kernel(key,r,s,order)
        for i in range(3):forces[i].append(K*q*charge*field[i])
    if energy:
        for i,(qi,ri) in enumerate(local):
            for qj,rj in local[i+1:]:direct.append(K*qi*qj/norm(sub(ri,rj)))
            for qj,rj in local:
                H,_=_kernel(key,ri,rj,order);induced.append(.5*K*qi*qj*H)
    if key.backend=='sphere' and key.shells[0][2] is not None:
        a,_,Q=key.shells[0];S=fsum(qj/norm(rj) for qj,rj in local)
        for i in range(3):forces[i].append(K*q*(Q+a*S)*r[i]/norm(r)**3)
        induced.append(.5*K*a*S*S);background=K*Q*S
    elif key.backend in ('cavity','nested') and energy:
        total=fsum(qj for qj,_ in local)
        induced.append(.5*total*_surface_voltage(key,total,False))
        background=total*_surface_voltage(key,0.,True)
    elif key.backend=='plates':
        slope=(key.voltages[1]-key.voltages[0])/key.gap
        for i in range(3):forces[i].append(-q*slope*key.normal[i])
        if energy:background=fsum(qj*(key.voltages[0]+slope*dot(rj,key.normal)) for qj,rj in local)
    return tuple(fsum(c) for c in forces),fsum((*direct,*induced,background))


def _row(key,time,p,v,order):
    F,U=force_energy(key,p,order=order);kinetic=.5*key.mass*norm(v)**2
    return dict(time_s=time,position_m=p,velocity_m_s=v,force_n=F,
        acceleration_m_s2=tuple(x/key.mass for x in F),kinetic_energy_j=kinetic,
        interaction_energy_j=U,total_energy_j=fsum((kinetic,U)))


@lru_cache(maxsize=48)
def trajectory(case_id,substeps=128,order=256):
    key=KEYS[case_id]
    if not key.backend:raise ValueError('refused task has no reference trajectory')
    p=key.sources[key.moving][1];v=key.velocity
    a=tuple(x/key.mass for x in force_energy(key,p,order=order,energy=False)[0])
    rows=[_row(key,0.,p,v,order)];dt=key.duration/(20*substeps)
    for output in range(1,21):
        for _ in range(substeps):
            pn=tuple(p[i]+v[i]*dt+.5*a[i]*dt*dt for i in range(3))
            an=tuple(x/key.mass for x in force_energy(key,pn,order=order,energy=False)[0])
            vn=tuple(v[i]+.5*(a[i]+an[i])*dt for i in range(3))
            p,v,a=pn,vn,an
        rows.append(_row(key,output*key.duration/20,p,v,order))
    return tuple(rows)


def convergence(case_id):
    a,b=trajectory(case_id,64),trajectory(case_id,128)
    result=dict(position_difference_m=max(norm(sub(x['position_m'],y['position_m'])) for x,y in zip(a,b)),
                velocity_difference_m_s=max(norm(sub(x['velocity_m_s'],y['velocity_m_s'])) for x,y in zip(a,b)))
    if KEYS[case_id].backend=='plates':
        # Bound the separate image-series truncation, independently of time step.
        key=KEYS[case_id];errors=[(force_energy(key,r['position_m'],order=256),
                                 force_energy(key,r['position_m'],order=512)) for r in b]
        result.update(image_force_difference_n=max(norm(sub(x[0],y[0])) for x,y in errors),
                      image_energy_difference_j=max(abs(x[1]-y[1]) for x,y in errors))
    return result
