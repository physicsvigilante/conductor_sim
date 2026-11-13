"""M12a physics, authorability, strict scope and reusable viewer tests."""
from math import isfinite, pi
import json
import pytest

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, FixedNetCharge, FixedPotential,
    Femm5PlaneBackend, Grounded, GroundedPlaneBackend, ParallelPlateBackend,
    PointCharge, SolidSphere, SphericalShell,
)
from conductor_sim.model import EPSILON_0
from conductor_sim.solver import SingularField, UnsupportedProblem
from conductor_sim.visualization import SliceSpec, sample_sweep
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.viewer import render_html


Q = 1e-9
D = .2


def capacitor(charges=(), left=Grounded(), right=Grounded(), *, d=D):
    return ElectrostaticProblem(
        tuple(charges),
        (ConductingPlane(constraint=left,name="lower"),
         ConductingPlane(point_m=(0.,0.,d),normal=(0.,0.,-1.),
                         constraint=right,name="upper")),
    )


def test_two_fixed_potentials_produce_exact_linear_field_and_infinite_charge_totals():
    p=capacitor(left=FixedPotential(12.),right=FixedPotential(-6.))
    b=ParallelPlateBackend()
    assert b.supports(p)
    s=b.solve(p)
    assert s.sources==()
    assert s.potential_at((.3,2.,.0))==12.
    assert s.potential_at((.3,2.,D))==-6.
    for z in (.02,.10,.16):
        assert s.potential_at((4.,-7.,z))==pytest.approx(12.-90.*z)
        assert s.electric_field_at((4.,-7.,z))==pytest.approx((0.,0.,90.))
    assert s.potential_at((2.,0.,-.1))==12.
    assert s.potential_at((2.,0.,.3))==-6.
    assert s.electric_field_at((2.,0.,-.1))==(0.,0.,0.)
    assert s.electric_field_at((2.,0.,.3))==(0.,0.,0.)
    with pytest.raises(SingularField):
        s.electric_field_at((0.,0.,D))
    with pytest.raises(SingularField):
        s.electric_field_at((0.,0.,0.))
    for r in s.conductor_results:
        assert r.net_charge_c is None and r.outer_surface_charge_c is None
        assert r.inner_surface_charge_c==0.
    for x,y in ((0.,0.),(.3,-.7)):
        assert s.surface_charge_density_at('outer',(x,y,0.),conductor_name='lower')==pytest.approx(90*EPSILON_0)
        assert s.surface_charge_density_at('outer',(x,y,D),conductor_name='upper')==pytest.approx(-90*EPSILON_0)


def test_charge_free_zero_bias_and_common_gauge_shift():
    reference=ParallelPlateBackend().solve(capacitor())
    shifted=ParallelPlateBackend().solve(capacitor(left=FixedPotential(15.),right=FixedPotential(15.)))
    for z in (.01,.1,.19):
        point=(.02,.04,z)
        assert reference.potential_at(point)==0.
        assert shifted.potential_at(point)==pytest.approx(15.)
        assert shifted.electric_field_at(point)==(0.,0.,0.)
    assert all(r.net_charge_c==0 for r in shifted.conductor_results)
    for name,z in [('lower',0.),('upper',D)]:
        assert shifted.surface_charge_density_at('outer',(0.,0.,z),conductor_name=name)==0.


def test_one_source_is_symmetric_between_two_grounded_plates():
    p=capacitor((PointCharge(Q,(0.,0.,D/2)),))
    b=ParallelPlateBackend(image_order=64)
    s=b.solve(p)
    results=s.conductor_results
    assert [r.name for r in results]==['lower','upper']
    assert [r.net_charge_c for r in results]==pytest.approx([-Q/2,-Q/2])
    assert sum(r.net_charge_c for r in results)==pytest.approx(-Q)
    x=.055
    assert s.potential_at((x,0.,.07))==pytest.approx(s.potential_at((x,0.,.13)),rel=1e-4)
    assert s.potential_at((x,0.,D/2))>0
    assert abs(s.electric_field_at((x,0.,D/2))[2])<.03
    sig0=s.surface_charge_density_at('outer',(0.,0.,0.),conductor_name='lower')
    sig1=s.surface_charge_density_at('outer',(0.,0.,D),conductor_name='upper')
    assert sig0<0 and sig1<0
    assert sig0==pytest.approx(sig1,rel=1e-4)
    with pytest.raises(SingularField):
        s.potential_at((0.,0.,D/2))
    with pytest.raises(SingularField):
        s.electric_field_at((0.,0.,D/2))


def test_arbitrary_lateral_sources_grouping_and_exact_induced_excess_totals():
    a=PointCharge(Q,(.04,-.02,.05))
    b=PointCharge(-.6*Q,(-.05,.03,.14))
    p=capacitor((a,b,PointCharge(Q,(.04,-.02,.05)),PointCharge(-Q,(.04,-.02,.05))))
    solver=ParallelPlateBackend(image_order=64)
    s=solver.solve(p)
    assert len(s.sources)==2 and solver.supports(p)
    qa=-Q*(1-.05/D)+.6*Q*(1-.14/D)
    qb=-Q*.05/D+.6*Q*.14/D
    assert [r.net_charge_c for r in s.conductor_results]==pytest.approx((qa,qb))
    assert sum(r.net_charge_c for r in s.conductor_results)==pytest.approx(-.4*Q)
    assert abs(s.surface_charge_density_at('outer',(.04,-.02,0.),conductor_name='lower'))>0
    pt=(.025,.01,.11)
    singles=[solver.solve(capacitor((c,))) for c in (a,b)]
    assert s.potential_at(pt)==pytest.approx(sum(t.potential_at(pt) for t in singles),rel=1e-9,abs=1e-8)
    assert s.electric_field_at(pt)==pytest.approx(
        tuple(sum(t.electric_field_at(pt)[j] for t in singles) for j in range(3)),rel=1e-9,abs=1e-5)


def test_field_is_negative_spatial_gradient_and_series_converges():
    p=capacitor((PointCharge(Q,(.02,-.01,.075)),))
    s=ParallelPlateBackend(image_order=48).solve(p)
    s2=ParallelPlateBackend(image_order=96).solve(p)
    x=(.085,.0,.092)
    for i in range(3):
        step=1e-5
        plus=tuple(x[j]+(step if i==j else 0) for j in range(3))
        minus=tuple(x[j]-(step if i==j else 0) for j in range(3))
        gradient=-(s.potential_at(plus)-s.potential_at(minus))/(2*step)
        assert s.electric_field_at(x)[i]==pytest.approx(gradient,rel=2e-5,abs=1e-3)
    assert s.potential_at(x)==pytest.approx(s2.potential_at(x),abs=.01)
    assert s.electric_field_at(x)==pytest.approx(s2.electric_field_at(x),abs=.1)


def test_world_translation_and_rotation_invariance():
    p=capacitor((PointCharge(Q,(.04,.03,.06)),),left=FixedPotential(-3.),right=FixedPotential(5.))
    s=ParallelPlateBackend().solve(p)
    def tr(p): return (2.+p[2],-1.+p[1],.5+p[0])
    rotated=ElectrostaticProblem(
        (PointCharge(Q,tr((.04,.03,.06))),),
        (ConductingPlane(tr((0.,0.,0.)),(1.,0.,0.),FixedPotential(-3.),'lower'),
         ConductingPlane(tr((0.,0.,D)),(-1.,0.,0.),FixedPotential(5.),'upper')),
    )
    rs=ParallelPlateBackend().solve(rotated)
    for x in ((.025,.04,.11),(-.07,.0,.08)):
        assert rs.potential_at(tr(x))==pytest.approx(s.potential_at(x),rel=1e-10,abs=1e-6)
        e=s.electric_field_at(x)
        assert rs.electric_field_at(tr(x))==pytest.approx((e[2],e[1],e[0]),rel=1e-10,abs=1e-4)
    for name,z in [('lower',0.),('upper',D)]:
        point=(.01,-.02,z)
        assert rs.surface_charge_density_at('outer',tr(point),conductor_name=name)==pytest.approx(
            s.surface_charge_density_at('outer',point,conductor_name=name),rel=1e-10)


def test_strict_geometry_and_source_guards_keep_femm_out_of_unsupported_scope():
    b=ParallelPlateBackend()
    malformed=[
        ElectrostaticProblem(),
        ElectrostaticProblem(conductors=(ConductingPlane(),)),
        ElectrostaticProblem(conductors=(ConductingPlane(),ConductingPlane(point_m=(0.,0.,D)))),
        ElectrostaticProblem(conductors=(ConductingPlane(),ConductingPlane(point_m=(0.,0.,D),normal=(.01,0.,-1.)))),
        ElectrostaticProblem(conductors=(ConductingPlane(),ConductingPlane(point_m=(0.,0.,-D),normal=(0.,0.,-1.)))),
        ElectrostaticProblem(conductors=(ConductingPlane(),ConductingPlane(point_m=(0.,0.,0.),normal=(0.,0.,-1.)))),
        capacitor((PointCharge(Q,(0.,0.,0.)),)),
        capacitor((PointCharge(Q,(0.,0.,D)),)),
        capacitor((PointCharge(Q,(0.,0.,-.01)),)),
        capacitor((PointCharge(Q,(0.,0.,D+.01)),)),
        capacitor((PointCharge(Q,(0.,0.,-.01)),PointCharge(-Q,(0.,0.,-.01)))),
        capacitor(tuple(PointCharge(Q,(0.,0.,.01+.005*i)) for i in range(33))),
    ]
    for p in malformed:
        assert not b.supports(p)
        with pytest.raises(UnsupportedProblem):
            b.solve(p)
    assert GroundedPlaneBackend().supports(ElectrostaticProblem(
        (PointCharge(Q,(0.,0.,.05)),),(ConductingPlane(),)))
    fp=ElectrostaticProblem((PointCharge(Q,(0.,0.,.05)),),
                            (ConductingPlane(constraint=FixedPotential(0.)),))
    assert not GroundedPlaneBackend().supports(fp)
    with pytest.raises(UnsupportedProblem):
        GroundedPlaneBackend().solve(fp)
    started=[]
    femm=Femm5PlaneBackend(session_factory=lambda:started.append('started'))
    assert not femm.supports(capacitor())
    with pytest.raises(UnsupportedProblem):
        femm.solve(capacitor())
    assert not started


def test_model_constraint_validation_and_image_series_parameter_contract():
    with pytest.raises(TypeError):
        ConductingPlane(constraint=FixedNetCharge(0.))
    with pytest.raises(TypeError):
        SphericalShell(.1,.2,FixedPotential(3.))
    with pytest.raises(TypeError):
        SolidSphere(.1,FixedPotential(3.))
    for v in (float('nan'),float('inf'),float('-inf')):
        with pytest.raises(ValueError):
            FixedPotential(v)
    for n in (True,0,513,2.5):
        with pytest.raises(ValueError):
            ParallelPlateBackend(image_order=n)
    s=ParallelPlateBackend().solve(capacitor())
    with pytest.raises(ValueError):
        s.surface_charge_density_at('inner',(0.,0.,0.),conductor_name='lower')
    with pytest.raises(ValueError):
        s.surface_charge_density_at('outer',(0.,0.,.01),conductor_name='lower')
    with pytest.raises(ValueError):
        s.surface_charge_density_at('outer',(0.,0.,0.),conductor_name='missing')


def test_two_planes_use_existing_slice_geometry_density_and_json_viewer():
    neutral=capacitor(left=FixedPotential(0.),right=FixedPotential(20.))
    sources=capacitor((PointCharge(Q,(0.,0.,.06)),PointCharge(-.5*Q,(0.,0.,.14))))
    meridian=SliceSpec.meridian(neutral,resolution=15)
    assert meridian.y_range_m[1]>D
    bundle=sample_sweep([('biased',neutral),('charged',sources)],
                        ParallelPlateBackend(image_order=24),plane=meridian,
                        seeds=[],line_steps=100)
    assert len(bundle['frames'])==2
    for frame in bundle['frames']:
        assert len(frame['geometry'])==2
        assert all(g['kind']=='plane' and g['visible'] for g in frame['geometry'])
        assert len(frame['profiles'])==2
        assert [r['conductor'] for r in frame['profiles']]==['lower','upper']
        assert len(frame['conductor_results'])==2
        assert any(region=='vacuum' for row in frame['grid']['region'] for region in row)
        assert any(region=='metal' for row in frame['grid']['region'] for region in row)
    assert bundle['frames'][0]['conductor_results'][0]['net_charge_c'] is None
    assert bundle['frames'][1]['conductor_results'][0]['net_charge_c'] is not None
    assert context_for_solution(neutral,ParallelPlateBackend().solve(neutral)).density_at is not None
    assert "n/a" in render_html(bundle)
    json.dumps(bundle,allow_nan=False)
