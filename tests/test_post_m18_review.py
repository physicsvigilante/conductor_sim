"""Regressions reproduced against accepted M18a, independent of the new viewer."""
from dataclasses import replace
import json
from math import hypot
import pytest
from conductor_sim import (ConductingPlane, ElectrostaticProblem, GroundedPlaneBackend,
    PointCharge, Motion1DSpec, Motion2DSpec, integrate_motion_1d, integrate_motion_2d)
from conductor_sim._motion_validation import validate_planar_motion


def problem():
    return ElectrostaticProblem((PointCharge(1e-12,(0.,0.,.1)),),(ConductingPlane(),))


@pytest.mark.parametrize('dimension',[1,2])
@pytest.mark.parametrize('duration,interval',[(.07,.01),(.14,.02),(.035,.005)])
def test_output_grid_has_one_exact_final_time(dimension,duration,interval):
    if dimension==1:
        spec=Motion1DSpec(problem(),0,1e-6,(0,0,1),0.,duration,interval)
        result=integrate_motion_1d(spec,GroundedPlaneBackend())
    else:
        spec=Motion2DSpec(problem(),0,1e-6,(.01,0,0),duration,interval)
        result=integrate_motion_2d(spec,GroundedPlaneBackend())
    times=[s.time_s for s in result.samples]
    assert len(times)==8
    assert times[-1]==duration
    assert all(a<b for a,b in zip(times,times[1:]))


def saved():
    spec=Motion2DSpec(problem(),0,1e-6,(.01,0,0),.02,.01)
    return json.loads(json.dumps(integrate_motion_2d(spec,GroundedPlaneBackend()).as_dict()))


def test_saved_superluminal_velocity_cannot_masquerade_as_classical_motion():
    data=saved()
    row=data['samples'][1]
    row['velocity_m_s']=[3e8,0,0]
    row['kinetic_energy_j']=.5*data['mass_kg']*hypot(*row['velocity_m_s'])**2
    row['total_energy_j']=row['kinetic_energy_j']+row['interaction_energy_j']
    with pytest.raises(ValueError,match='speed'):
        validate_planar_motion(data)


@pytest.mark.parametrize('guard',[0.,-1.,True])
def test_saved_surface_guard_is_positive_finite_metadata(guard):
    data=saved();data['guard_m']=guard
    with pytest.raises(ValueError,match='guard'):
        validate_planar_motion(data)


@pytest.mark.parametrize('duration,interval',[(.071,.01),(2e-8,3e-9),(1.,1.),(.14,.14/2000)])
def test_shared_time_grid_is_bounded_and_keeps_fractional_final_interval(duration,interval):
    from conductor_sim._motion_time import output_times
    times=output_times(duration,interval)
    assert 1<=len(times)<=2000
    assert times[-1]==duration and len(times)==len(set(times))
    assert all(a<b for a,b in zip((0.,*times),times))
    if duration==.071:assert times[-2]==pytest.approx(.07)


def test_output_grid_rejects_overflowing_requested_interval_count():
    from conductor_sim._motion_time import output_times
    with pytest.raises(ValueError,match='2000'):
        output_times(1e300,1e-300)
