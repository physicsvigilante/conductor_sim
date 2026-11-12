"""M3 geometry-compiler and API contract, with a strict fake FEMM client."""
from math import pi
import pytest

from conductor_sim import ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell
from conductor_sim.femm5 import Femm5Backend, Femm5Settings, _SOURCE_CORE
from conductor_sim.solver import UnsupportedProblem

from test_femm5_adapter import FakeClient, FakeSession, calls


def problem(position=(0,0,.04), constraint=Grounded(), q=1e-9):
    return ElectrostaticProblem(charges=(PointCharge(q,position),),
        conductors=(SphericalShell(.1,.12,constraint),))


def fake_backend(settings=None):
    client=FakeClient()
    session=FakeSession(client)
    return Femm5Backend(settings=settings,session_factory=lambda:session),client,session


def test_offset_sphere_is_axisymmetric_with_one_physical_shell():
    backend,client,session=fake_backend()
    with backend.solve(problem()) as solution:
        assert solution.source_radius_m==pytest.approx(.01)
        assert solution.source_displacement_m==pytest.approx(.04)
        assert solution.symmetry_axis==(0,0,1)
    arcs=calls(client,"ei_addarc")
    assert len(arcs)==8
    assert (0,.03,.01,.04,90,5) == pytest.approx(arcs[-2])
    assert (.01,.04,0,.05,90,5) == pytest.approx(arcs[-1])
    assert len(calls(client,"ei_addsegment"))==7
    props=calls(client,"ei_setarcsegmentprop")
    assert sum(arg[4]=="shell" for arg in props)==4
    assert sum(arg[4]==_SOURCE_CORE for arg in props)==2
    assert len(calls(client,"ei_addblocklabel"))==4
    assert calls(client,"ei_addpointprop")==[]
    assert session.opens==session.closes==1


def test_world_axis_rotation_maps_coordinates_and_field():
    backend,client,session=fake_backend()
    with backend.solve(problem(position=(.04,0,0))) as sol:
        assert sol.symmetry_axis==pytest.approx((1,0,0))
        assert sol.potential_at((.065,.03,0))==12
        assert calls(client,"eo_getpointvalues")[-1]==pytest.approx((.03,.065))
        # Fake FEMM E_r=4, E_z=6. With physical axis x,
        # the radial component maps onto y, and axial onto x.
        assert sol.electric_field_at((.065,.03,0)) == pytest.approx((6,4,0))
        with pytest.raises(UnsupportedProblem, match="source core"):
            sol.potential_at((.04,0,0))
    assert session.closes==1


def test_surface_density_is_from_numerical_field_not_analytical_charge():
    backend,client,_=fake_backend()
    eps=8.8541878128e-12
    with backend.solve(problem()) as sol:
        # θ=0 gives E_z=6 V/m from fake FEMM; θ=π/2 gives E_r=4.
        assert sol.surface_charge_density("inner",0)==pytest.approx(-6*eps)
        assert sol.surface_charge_density("inner",pi/2)==pytest.approx(-4*eps)
        assert sol.surface_charge_density("outer",pi/2)==pytest.approx(4*eps)
        for s,angle in [("middle",0),("inner",-1),("inner",4)]:
            with pytest.raises(ValueError):
                sol.surface_charge_density(s, angle)
        with pytest.raises(ValueError):
            sol.surface_charge_density("inner",.5,offset_fraction=.2)


def test_source_near_wall_limits_core_to_clearance():
    backend,client,_=fake_backend()
    with backend.solve(problem(position=(0,0,.092))) as sol:
        assert sol.source_radius_m==pytest.approx(.002)
        assert sol.source_displacement_m==pytest.approx(.092)
        with pytest.raises(UnsupportedProblem):
            sol.potential_at((0,0,.092))
    # Source sphere ends at 0.094 < a=0.1


def test_multi_source_locations_and_outside_cavity_rejected():
    backend,_,session=fake_backend()
    shell=problem().conductors
    bad=[
        ElectrostaticProblem(charges=(PointCharge(1e-9,(0,0,.03)),PointCharge(2e-9,(0,0,.04))),conductors=shell),
        ElectrostaticProblem(charges=(PointCharge(1e-9,(.12,0,0)),), conductors=shell),
    ]
    for p in bad:
        assert not backend.supports(p)
        with pytest.raises(UnsupportedProblem):
            backend.solve(p)
    assert session.opens==0


def test_coincident_offcenter_authored_charges_are_composed():
    p=ElectrostaticProblem(charges=(PointCharge(.75e-9,(0,0,.04)),PointCharge(.25e-9,(0,0,.04))),
                           conductors=problem().conductors)
    backend,client,_=fake_backend()
    assert backend.supports(p)
    with backend.solve(p):
        assert calls(client,"ei_addconductorprop")[-1] == (_SOURCE_CORE,0,1e-9,0)
