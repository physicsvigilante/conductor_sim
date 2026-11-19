"""Post-M40b audit regressions. No new electrostatic regime is claimed."""
from copy import deepcopy
from dataclasses import replace
from math import hypot, pi, sqrt, ulp
import random

import pytest

from conductor_sim.model import (
    ConductingPlane, ElectrostaticProblem, Grounded, PointCharge, SphericalShell,
)
from conductor_sim.grounded_plane import GroundedPlaneBackend
from conductor_sim.motion_1d import Motion1DSpec
from conductor_sim.motion_2d import _unit as motion_plane_unit
from conductor_sim.visualization import SliceSpec, unit as slice_unit
from conductor_sim.spherical_multicavity import solve_spherical_sigma
from conductor_sim.spherical_sigma_atlas import surface_atlas_manifest
from tools.m40a_spherical_example import teacher_model
from tools.m40b_spherical_atlas_reference import check_manifest


TINY = ulp(0.0)


def _motion_axis(raw):
    problem = ElectrostaticProblem(
        charges=(PointCharge(1e-9),),
        conductors=(SphericalShell(1., 2., Grounded()),),
    )
    return Motion1DSpec(problem, 0, 1., raw, 0., .01, .005).axis


@pytest.mark.parametrize('direction', [
    (TINY, TINY, 0.), (2*TINY, -TINY, 3*TINY), (0., TINY, -TINY),
])
@pytest.mark.parametrize('consumer', [
    lambda v: ConductingPlane(normal=v).normal,
    _motion_axis, slice_unit, motion_plane_unit,
])
def test_subnormal_ratios_define_a_true_unit_direction(consumer, direction):
    scaled = tuple(v / max(map(abs, direction)) for v in direction)
    expected = tuple(v / hypot(*scaled) for v in scaled)
    got = consumer(direction)
    assert got == pytest.approx(expected, rel=3e-16, abs=0.)
    assert hypot(*got) == pytest.approx(1., abs=3e-16)


def test_slice_projection_roundtrip_and_physical_metre_scale():
    plane = SliceSpec(u=(TINY,TINY,0.), v=(0.,0.,1.))
    point = plane.world((.25,-.125))
    assert hypot(*point) == pytest.approx(hypot(.25,.125), rel=4e-16)
    assert plane.project(point) == pytest.approx((.25,-.125,0.), rel=4e-16, abs=1e-16)


@pytest.mark.parametrize('charge', [1e-9, -2.3e-9])
def test_subnormal_plane_matches_independent_mirror_potential_field_force(charge):
    # Explicit independent geometry: normal=(1,1,0)/sqrt(2), source=(1,1,0),
    # reflected charge at (-1,-1,0); no production mirror helper is used.
    plane = ConductingPlane(normal=(TINY,TINY,0.))
    problem = ElectrostaticProblem(charges=(PointCharge(charge,(1.,1.,0.)),),
                                   conductors=(plane,))
    result = GroundedPlaneBackend().solve(problem)
    point=(2.,2.,1.); direct=(1.,1.,1.); image=(3.,3.,1.)
    rd, ri = hypot(*direct), hypot(*image)
    k=1/(4*pi*problem.permittivity_f_per_m)
    expected_v = k*charge*(1/rd-1/ri)
    expected_e = tuple(k*charge*(a/rd**3-b/ri**3) for a,b in zip(direct,image))
    expected_f = tuple(-k*charge*charge*x/hypot(2.,2.,0.)**3 for x in (2.,2.,0.))
    assert result.potential_at(point) == pytest.approx(expected_v,rel=2e-14,abs=0.)
    assert result.electric_field_at(point) == pytest.approx(expected_e,rel=2e-14,abs=0.)
    assert result.induced_force_on_charge_n == pytest.approx(expected_f,rel=2e-14,abs=0.)
    # Sigma at the plane origin from a separately expressed image solution.
    expected_sigma = -charge*sqrt(2)/(2*pi*hypot(1.,1.,0.)**3)
    assert result.surface_charge_density_at('outer',(0.,0.,0.)) == pytest.approx(expected_sigma,rel=2e-14,abs=0.)


def test_normalization_keeps_ordinary_arithmetic_bit_for_bit():
    rng=random.Random(104)
    for _ in range(100):
        v=tuple(rng.uniform(-10,10) for _ in range(3))
        expected=tuple(x/hypot(*v) for x in v)
        for got in (ConductingPlane(normal=v).normal, slice_unit(v), motion_plane_unit(v), _motion_axis(v)):
            assert got == expected


@pytest.mark.parametrize('v', [(0.,0.,0.), (float('inf'),0.,0.),
                               (float('nan'),1.,0.), (1.7e308,1.7e308,0.)])
def test_direction_refusals_are_not_relaxed(v):
    for consumer in (lambda x:ConductingPlane(normal=x),slice_unit,motion_plane_unit,_motion_axis):
        with pytest.raises(ValueError):
            consumer(v)


def _atlas(last_charged=False):
    model=teacher_model()
    if last_charged:
        order=(0,2,1)
        model=replace(model,
            geometry=replace(model.geometry,voids=tuple(model.geometry.voids[i] for i in order)),
            gauss_assembly=replace(model.gauss_assembly,cavities=tuple(model.gauss_assembly.cavities[i] for i in order)))
    grounded=replace(model,gauss_assembly=replace(model.gauss_assembly,constraint=Grounded()))
    return surface_atlas_manifest(solve_spherical_sigma(model),solve_spherical_sigma(grounded),directions=((1.,0.,0.),))


@pytest.mark.parametrize('last_charged',[False,True])
def test_atlas_cannot_drop_a_native_cavity_and_its_face_witness(last_charged):
    doc=deepcopy(_atlas(last_charged))
    doc['cavities'].pop()
    for state in ('isolated','grounded'):
        doc['native_m40a_v2_witness_records'][state].pop(-2)
    with pytest.raises(ValueError,match='cavit'):
        check_manifest(doc)


def test_complete_atlas_is_still_accepted():
    result=check_manifest(_atlas())
    assert result['cavities_checked'] == 3
    assert result['sources_checked'] == 4
