"""Teacher-authored fixed-axis dipole motion, not inferred from workbook Q47."""
from __future__ import annotations
import argparse
from math import pi
from pathlib import Path
from conductor_sim.dipoles import RigidDipole, DipoleModel
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.prescribed_fields import (
    UniformElectricField, FixedPointField, UniformRodField, UniformSphereField)
from conductor_sim.dipole_motion import DipoleMotionSpec, integrate_dipole_motion

CASES = ('small-oscillation', 'libration', 'rotation', 'point-pivot', 'mixed-sources', 'source-guard')

def make_case(case):
    d = RigidDipole(1e-9, .04, pivot_m=(0.,0.,0.))
    sources = (NamedSource('field:uniform','Uniform E', UniformElectricField((120.,0.,0.))),)
    angle, velocity, duration, interval, guard = .1, 0., 1.5, .01, 1e-5
    if case == 'libration':
        angle, velocity = 0., 15.
    elif case == 'rotation':
        angle, velocity = 0., 25.
    elif case == 'point-pivot':
        d = RigidDipole(1e-9,.04,pivot_m=(-.015,0.,0.))
        sources = (NamedSource('point:held','Held point source',FixedPointField(2e-9,(.12,.06,0.))),)
        angle, velocity, duration, interval = .4, 1., .5, .005
    elif case == 'mixed-sources':
        sources = (
            NamedSource('rod:held','Held finite rod',UniformRodField(.3e-9,(-.09,.12,0.),(.07,.14,0.))),
            NamedSource('sphere:volume','Fixed volume charge',UniformSphereField(.4e-9,.03,(.15,-.08,0.))),
            NamedSource('field:uniform','Uniform E',UniformElectricField((90.,25.,0.))),
        )
        angle, velocity, duration, interval = .7, -3., 1., .01
    elif case == 'source-guard':
        sources = (NamedSource('point:near','Held singular point',FixedPointField(-1e-13,(.02,0.,0.))),)
        angle, velocity, duration, interval, guard = -.3, 10., .12, .005, .001
    elif case != 'small-oscillation':
        raise ValueError('unknown motion case')
    model = DipoleModel(d, sources, model_id='m23b:'+case,
        provenance='Teacher hypothetical: explicit fixed ideal hinge, inertia and angular initial conditions; not supplied by workbook Q47.')
    return DipoleMotionSpec(model=model, rotation_axis=(0.,0.,1.), inertia_kg_m2=4.8e-11,
        initial_angle_rad=angle, initial_angular_velocity_rad_s=velocity,
        duration_s=duration, sample_interval_s=interval, source_guard_m=guard)

def main():
    from conductor_sim.dipole_motion_viewer import write_dipole_motion
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=CASES,default=CASES[0])
    ap.add_argument('--output',type=Path,default=Path('output/m23b-motion.html'))
    a=ap.parse_args()
    result=integrate_dipole_motion(make_case(a.case))
    write_dipole_motion(result,a.output,title='M23b · '+a.case.replace('-',' '))
    print(a.case, result.termination, len(result.samples), 'samples;',a.output)
if __name__=='__main__':main()
