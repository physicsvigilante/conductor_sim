"""Regression cases reproduced against the accepted M17b tree before repair."""
from copy import deepcopy
from dataclasses import replace
import pytest
from conductor_sim import (ElectrostaticProblem, PointCharge, SolidSphere, Grounded,
    ExternalSphereBackend, Motion2DSpec, integrate_motion_2d, sample_motion_2d)
from conductor_sim.motion_viewer_2d import motion_slice_2d
from conductor_sim.viewer import render_html


@pytest.mark.parametrize('position,velocity', [
    ((.1,.2,.3),(.01,.02,.03)),
    ((.18,.12,.06),(.03,.02,.01)),
    ((.1,.2,.3),(-.01,-.02,-.03)),
    ((.1,.2,.3),(.01,.02,.030000000000000002)),
])
def test_roundoff_in_radial_launch_must_not_invent_parallel_slice_axes(position,velocity):
    p=ElectrostaticProblem((PointCharge(1e-9,position),),(SolidSphere(.01,Grounded()),))
    spec=Motion2DSpec(p,0,1e-6,velocity,.001,.0005)
    result=integrate_motion_2d(spec,ExternalSphereBackend())
    plane=motion_slice_2d(spec,result,resolution=9)
    assert abs(sum(x*y for x,y in zip(plane.u,plane.v)))<1e-12
    assert all(abs(plane.project(row.position_m)[2])<1e-12 for row in result.samples)


@pytest.fixture(scope='module')
def motion_bundle():
    p=ElectrostaticProblem((PointCharge(1e-9,(.18,0.,0.)),),
                          (SolidSphere(.07,Grounded()),))
    spec=Motion2DSpec(p,0,1e-6,(.015,.13,.07),.04,.005)
    r=integrate_motion_2d(spec,ExternalSphereBackend())
    b=sample_motion_2d(spec,ExternalSphereBackend(),result=r,
        resolution=9,line_steps=8,seeds=[],profile_samples=9,visual_frames=3)
    return spec,r,b


@pytest.mark.parametrize('mutation',[
    'charge-sign','mass','graph-position','displacement','velocity','acceleration',
    'kinetic','total','drift-max','plane-norm','slice-basis','counts','event-time',
    'event-position','unknown-termination','out-of-plane-force',
])
def test_all_graph_rows_and_metadata_are_validated_not_just_field_frames(motion_bundle,mutation):
    spec,r,b=motion_bundle
    bad=deepcopy(b);m=bad['motion_2d'];row=m['samples'][1]
    assert 1 not in m['frame_sample_indices']
    if mutation=='charge-sign':bad['frames'][0]['charges'][0]['charge_c']*=-1
    elif mutation=='mass':m['mass_kg']*=2
    elif mutation=='graph-position':row['position_m']=[1e8,1e8,1e8]
    elif mutation=='displacement':row['displacement_m'][0]=999.
    elif mutation=='velocity':row['velocity_m_s'][0]=999.
    elif mutation=='acceleration':row['acceleration_m_s2'][0]=999.
    elif mutation=='kinetic':row['kinetic_energy_j']=999.
    elif mutation=='total':row['total_energy_j']=999.
    elif mutation=='drift-max':m['max_abs_energy_drift_j']=999.
    elif mutation=='plane-norm':m['plane_normal']=[2*x for x in m['plane_normal']]
    elif mutation=='slice-basis':
        for f in bad['frames']:f['plane']['u']=[2*x for x in b['frames'][0]['plane']['u']]
    elif mutation=='counts':m['graph_sample_count']+=1
    elif mutation=='event-time':m['events']=[dict(kind='surface-guard',time_s=-1.,position_m=row['position_m'])]
    elif mutation=='event-position':
        m['termination']='surface-guard'
        m['events']=[dict(kind='surface-guard',time_s=m['samples'][-1]['time_s'],position_m=[999.,0.,0.])]
    elif mutation=='unknown-termination':m['termination']='collision'
    else:
        row['force_n']=[row['force_n'][i]+1e-8*m['plane_normal'][i] for i in range(3)]
        row['acceleration_m_s2']=[v/m['mass_kg'] for v in row['force_n']]
    with pytest.raises(ValueError):render_html(bad)


def test_reject_bad_supplied_trajectory_before_any_electrostatic_solve(motion_bundle):
    spec,r,b=motion_bundle
    bad=replace(r,samples=(r.samples[0],replace(r.samples[1],kinetic_energy_j=999.),*r.samples[2:]))
    class MustNotSolve:
        def solve(self,problem):raise AssertionError('invalid saved data must not acquire solver resources')
    with pytest.raises(ValueError):
        sample_motion_2d(spec,MustNotSolve(),result=bad,visual_frames=3,
            resolution=9,line_steps=8,seeds=[],profile_samples=9)


def test_sparse_circular_orbit_never_draws_chord_through_metal():
    from math import pi, sqrt
    from conductor_sim import force_energy
    from conductor_sim.geometry import physical_boundaries
    p=ElectrostaticProblem((PointCharge(1e-9,(.1,0,0)),),(SolidSphere(.07,Grounded()),))
    backend=ExternalSphereBackend()
    f=force_energy(backend.solve(p)).charges[0].total_force_n[0]
    speed=sqrt(-f*.1/1e-6)
    duration=pi*.1/speed
    spec=Motion2DSpec(p,0,1e-6,(0,speed,0),duration,duration)
    result=integrate_motion_2d(spec,backend)
    assert result.termination=='duration' and len(result.samples)==2
    assert physical_boundaries(p.conductors[0])[0].first_hit(
        result.samples[0].position_m,result.samples[-1].position_m) is not None
    b=sample_motion_2d(spec,backend,result=result,visual_frames=2,
        resolution=9,line_steps=8,seeds=[],profile_samples=9)
    assert b['motion_2d']['trajectory_break_indices']==[1]
    assert 'breaks.has(i)' in render_html(b)
    # Also repair older exported bundles at rendering, without altering input.
    del b['motion_2d']['trajectory_break_indices']
    import json
    html=render_html(b)
    embedded=json.loads(html.split('<script id="simulation-data" type="application/json">')[1].split('</script>')[0])
    assert embedded['motion_2d']['trajectory_break_indices']==[1]
    assert 'trajectory_break_indices' not in b['motion_2d']
