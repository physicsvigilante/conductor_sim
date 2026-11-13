"""Independent M15c keys and Coulomb/image reference. No conductor_sim imports.

The cavity images are constructed explicitly in world coordinates. Uniform
voltages use sums of physical inner/outer surface potentials plus a small
independent Gaussian elimination for grounded net charges, not engine helpers.
"""
from dataclasses import dataclass
from math import fsum, hypot, pi

K=1/(4*pi*8.8541878128e-12)

@dataclass(frozen=True)
class Key:
    sources: tuple = ()             # (q_C, local xyz_m), authored order
    shells: tuple = ()              # (inner radius, outer radius, Q_C or None for ground)
    end: tuple = ()
    center: tuple = (0.,0.,0.)
    backend: str | None = None
    refusal_words: tuple = ()

KEYS = {
    'centre-outward':Key(((1.2e-9,(0,0,0)),),((.12,.15,None),),(0,0,.07),backend='single'),
    'negative-inward':Key(((-.7e-9,(.08,0,0)),),((.11,.14,None),),(.02,0,0),backend='single'),
    'translated-isolated':Key(((.9e-9,(.02,-.01,.01)),),((.12,.16,1.5e-9),),
                              (.06,.02,.04),(.8,-.4,.2),'single'),
    'signed-reordering':Key(((.8e-9,(-.03,0,-.03)),(-.45e-9,(.01,.03,.015))),
                            ((.1,.13,None),),(.04,0,.04),backend='single'),
    'nested-ground':Key(((1e-9,(0,0,-.04)),),((.12,.15,.4e-9),(.45,.50,None)),
                        (0,0,.08),backend='nested'),
    'inner-ground-shield':Key(((.6e-9,(0,0,.02)),),
            ((.1,.12,None),(.2,.22,-.3e-9),(.3,.32,.8e-9)),(0,0,.06),backend='nested'),
    # Independent Gauss-law result after floating connection: inner=-q,
    # outer=Q_initial_group+q. The wire is then removed before the sweep.
    'connect-disconnect':Key(((.6e-9,(0,0,.02)),),((.1,.13,-.6e-9),(.22,.25,.8e-9)),
                             (0,0,.07),backend='nested'),
    'zero-net-sources':Key(((1e-9,(0,0,-.05)),(-1e-9,(.03,0,0))),
                           ((.12,.15,.2e-9),),(0,0,.05),backend='single'),
    'cross-metal':Key(refusal_words=('metal','cavity')),
    'source-collision':Key(refusal_words=('source','singular')),
    'cavity-time-motion':Key(refusal_words=('time','velocity')),
    'numerical-cavity-force':Key(refusal_words=('femm','numerical','force')),
}

def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))

def norm(x):
    return hypot(*x)

def states_at(key,t):
    start=key.sources[0][1]
    moved=tuple(start[i]+t*(key.end[i]-start[i]) for i in range(3))
    return ((key.sources[0][0],moved),)+key.sources[1:]

def _surface_potentials(shells,net,q):
    """Reduced conductor voltages from physical uniform face charges."""
    values=[]
    for i,(_,b,_) in enumerate(shells):
        terms=[fsum((q,*net[:i+1]))/b]
        for j,(aj,bj,_) in enumerate(shells[i+1:],i+1):
            enclosed=fsum((q,*net[:j]))
            terms.extend((-enclosed/aj,fsum((enclosed,net[j]))/bj))
        values.append(fsum(terms))
    return values

def _uniform(shells,q,background):
    n=len(shells)
    net=[float(Q) if background and Q is not None else 0. for _,_,Q in shells]
    unknown=[i for i,(_,_,Q) in enumerate(shells) if Q is None]
    base=_surface_potentials(shells,net,q)
    columns=[]
    for j in unknown:
        unit=[0.]*n;unit[j]=1.
        columns.append(_surface_potentials(shells,unit,0.))
    matrix=[[columns[j][i] for j in range(len(unknown))]+[-base[i]] for i in unknown]
    for c in range(len(matrix)):
        pivot=max(range(c,len(matrix)),key=lambda i:abs(matrix[i][c]))
        matrix[c],matrix[pivot]=matrix[pivot],matrix[c]
        factor=matrix[c][c]
        matrix[c]=[x/factor for x in matrix[c]]
        for r in range(len(matrix)):
            if r!=c:
                f=matrix[r][c]
                matrix[r]=[a-f*b for a,b in zip(matrix[r],matrix[c])]
    for i,row in zip(unknown,matrix):
        net[i]=row[-1]
    return K*_surface_potentials(shells,net,q)[0]

def observe(key,t):
    """Selected source force, energy pieces and path-projected force, in SI."""
    sources=states_at(key,t)
    a=key.shells[0][0]
    forces=[[],[],[]];pair=[];polar=[]
    for i,(qi,x) in enumerate(sources):
        for j,(qj,y) in enumerate(sources):
            if i<j:
                r=norm(sub(x,y));pair.append(K*qi*qj/r)
            if i==0 and j!=0:
                delta=sub(x,y);r=norm(delta)
                for c in range(3):
                    forces[c].append(K*qi*qj*delta[c]/r**3)
            d=norm(y)
            if d==0:
                polar.append(-.5*K*qi*qj/a)
                continue
            image=tuple(a*a*yc/(d*d) for yc in y)
            qimage=-qj*a/d
            delta=sub(x,image);r=norm(delta)
            polar.append(.5*K*qi*qimage/r)
            if i==0:
                for c in range(3):
                    forces[c].append(K*qi*qimage*delta[c]/r**3)
    q=fsum(v for v,_ in sources)
    induced_v=_uniform(key.shells,q,False)
    background_v=_uniform(key.shells,0.,True)
    F=tuple(fsum(v) for v in forces)
    direct=fsum(pair)
    induced=fsum((*polar,.5*q*induced_v))
    background=q*background_v
    direction=sub(key.end,key.sources[0][1]);length=norm(direction)
    return {'force_n':F,'force_parallel_n':fsum(F[i]*direction[i]/length for i in range(3)),
            'energy_direct_j':direct,'energy_induced_j':induced,
            'energy_background_j':background,'energy_j':fsum((direct,induced,background))}

def work_integral(key,intervals=80):
    """Independent Simpson quadrature of -F·dr, not an energy subtraction."""
    if intervals<2 or intervals%2:
        raise ValueError('Simpson integration needs an even interval count')
    terms=[]
    for i in range(intervals+1):
        weight=1 if i in (0,intervals) else 4 if i%2 else 2
        terms.append(weight*observe(key,i/intervals)['force_parallel_n'])
    length=norm(sub(key.end,key.sources[0][1]))
    return -length*fsum(terms)/(3*intervals)
