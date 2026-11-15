"""M27a: independent Coulomb-line references, guards and static visualization."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from math import cos, fsum, hypot, isclose, pi, sin, sqrt
import json

import pytest

from conductor_sim import (
    EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformElectricField, UniformRingAxisField,
    UniformRingField,
)
from conductor_sim.solver import SingularField, UnsupportedProblem
from conductor_sim.visualization import SliceSpec
from conductor_sim.ring_field_observer import (
    sample_ring_field_slice, validate_ring_field_slice, render_ring_field_slice,
)
from conductor_sim.prescribed_viewer_validation import _restore_field

K = 1/(4*pi*EPSILON_0)


def reference_ring(charge, radius, center, axis, point, *, eps=EPSILON_0, n=8192):
    """Independent direct Coulomb midpoint-line integral, no elliptic API."""
    length=hypot(*axis)
    direction=tuple(v/length for v in axis)
    seed=(1.,0.,0.) if abs(direction[0])<.8 else (0.,1.,0.)
    f=sum(seed[i]*direction[i] for i in range(3))
    along=tuple(seed[i]-f*direction[i] for i in range(3))
    size=hypot(*along);u=tuple(t/size for t in along)
    v=(direction[1]*u[2]-direction[2]*u[1],
       direction[2]*u[0]-direction[0]*u[2],
       direction[0]*u[1]-direction[1]*u[0])
    V=0.; Ex=Ey=Ez=0.
    for i in range(n):
        t=2*pi*(i+.5)/n
        site=tuple(center[j]+radius*(cos(t)*u[j]+sin(t)*v[j]) for j in range(3))
        d=tuple(point[j]-site[j] for j in range(3))
        rr=hypot(*d)
        inv=1/rr
        V+=inv
        coeff=inv**3
        Ex+=d[0]*coeff; Ey+=d[1]*coeff; Ez+=d[2]*coeff
    multiplier=charge/(4*pi*eps*n)
    return multiplier*V,(multiplier*Ex,multiplier*Ey,multiplier*Ez)


def ring(**changes):
    opts=dict(charge_c=1.7e-9, radius_m=.2, center_m=(0.,0.,0.), axis=(0.,0.,1.))
    return UniformRingField(**(opts|changes))


def test_p1_old_axis_formulas_and_historical_offaxis_refusal():
    new=ring()
    old=UniformRingAxisField(new.charge_c,new.radius_m,new.center_m,new.axis)
    for z in (-2.,-.2,-.00001,0.,.00001,.2,2.):
        p=(0.,0.,z)
        potential=K*new.charge_c/hypot(.2,z)
        Ez=K*new.charge_c*z/hypot(.2,z)**3
        assert new.potential_at(p)==pytest.approx(potential,rel=2e-13)
        assert new.electric_field_at(p)==pytest.approx((0.,0.,Ez),rel=2e-13,abs=2e-17)
        assert old.potential_at(p)==pytest.approx(new.potential_at(p),rel=3e-14)
    assert new.electric_field_at((0,0,0))==(0.,0.,0.)
    with pytest.raises(UnsupportedProblem,match="symmetry axis"):
        old.electric_field_at((.06,0.,.11))
    with pytest.raises(FrozenInstanceError):
        new.radius_m=.3
    assert type(new) is UniformRingField and old.as_dict()["kind"]=="UniformRingAxisField"


@pytest.mark.parametrize("p",[
    (.035,.045,.04), (.12,-.033,.093),(.27,.04,-.052),
    (.21,0.,.035), (.03,-.19,-.015),(.42,.23,.32),
    (-.04,.018,.3),(.17,.03,-.22),
])
def test_p2_offaxis_full_vector_vs_independent_coulomb_integral(p):
    s=ring(charge_c=-2.5e-9,center_m=(.02,-.01,.03),axis=(0.,0.,1.))
    reference_v,reference_e=reference_ring(s.charge_c,s.radius_m,s.center_m,s.axis,p)
    assert s.potential_at(p)==pytest.approx(reference_v,rel=3e-11)
    assert s.electric_field_at(p)==pytest.approx(reference_e,rel=2e-10,abs=1e-11)
    # In the ring's true cylindrical coordinates azimuthal E is zero.
    rho_x,rho_y=p[0]-.02,p[1]+.01
    e=s.electric_field_at(p)
    assert e[0]*(-rho_y)+e[1]*rho_x==pytest.approx(0.,abs=1e-11)


@pytest.mark.parametrize("p",[(.003,0.,.12),(.0000001,0.,.15),(.05,.04,-.09),(.22,.03,.04),(.5,.2,.3)])
def test_p2_negative_gradient_and_nearaxis_stable(p):
    s=ring()
    h=1e-6
    expected=[]
    for j in range(3):
        left=list(p);right=list(p);left[j]-=h;right[j]+=h
        expected.append(-(s.potential_at(right)-s.potential_at(left))/(2*h))
    assert s.electric_field_at(p)==pytest.approx(expected,rel=1e-6,abs=1e-7)


@pytest.mark.parametrize("rho", [0.,1e-12,1e-9,1e-7,1e-5,1e-4])
def test_small_rho_analytic_radial_coefficient(rho):
    s=ring()
    z=.03;a=.2
    e=s.electric_field_at((rho,0,z))
    # Exact derivative of the axial-order rho^2 potential coefficient.
    reference_rho=K*s.charge_c*rho*(2*z*z-a*a)/(2*(a*a+z*z)**2.5)
    if rho==0:
        assert e[0]==0
    else:
        assert e[0]==pytest.approx(reference_rho,rel=1e-5,abs=1e-18)
    assert e[1]==0.


def test_p3_arbitrary_rotations_and_axis_sign_and_charge_reversal():
    original=ring(center_m=(.02,-.04,.03))
    # Rotate +90 degrees around world Y: (x,y,z)->(z,y,-x).
    rot=lambda v:(v[2],v[1],-v[0])
    shift=(.4,-.3,.2)
    changed=ring(center_m=tuple(rot(original.center_m)[i]+shift[i] for i in range(3)),
                 axis=rot(original.axis))
    for p in ((.1,.03,.1),(-.03,.05,.02),(.35,-.08,.33)):
        p2=tuple(rot(p)[i]+shift[i] for i in range(3))
        assert changed.potential_at(p2)==pytest.approx(original.potential_at(p),rel=2e-13)
        assert changed.electric_field_at(p2)==pytest.approx(rot(original.electric_field_at(p)),rel=4e-13,abs=5e-12)
    reversed_axis=ring(axis=(0.,0.,-1.))
    negative=ring(charge_c=-original.charge_c)
    q=(.08,.09,.07)
    assert reversed_axis.electric_field_at(q)==pytest.approx(ring().electric_field_at(q),rel=1e-13)
    assert reversed_axis.potential_at(q)==pytest.approx(ring().potential_at(q),rel=1e-13)
    assert negative.electric_field_at(q)==pytest.approx(tuple(-v for v in ring().electric_field_at(q)),rel=1e-13)
    assert negative.potential_at(q)==pytest.approx(-ring().potential_at(q),rel=1e-13)


def test_permittivity_and_superposition_far_field():
    s=ring()
    p=(2.,3.,8.)
    original=s.potential_at(p)
    assert s.potential_at(p,permittivity_f_m=1.35*EPSILON_0)==pytest.approx(original/1.35,rel=1e-14)
    assert s.potential_at((0,0,1e6))==pytest.approx(K*s.charge_c/1e6,rel=1e-13)
    companion=ring(charge_c=-1.2e-9,center_m=(.3,0.,.02),axis=(1.,0.,0.))
    point=FixedPointField(2e-9,(.0,.2,.1))
    uniform=UniformElectricField((12.,-9.,3.))
    field=PrescribedField((s,companion,point,uniform))
    assert field.potential_at(p)==pytest.approx(fsum(v.potential_at(p) for v in field.sources),rel=1e-13)
    assert field.electric_field_at(p)==pytest.approx(tuple(
        fsum(v.electric_field_at(p)[i] for v in field.sources) for i in range(3)),rel=1e-13)
    restored=_restore_field(json.loads(json.dumps(field.as_dict())))
    assert restored.as_dict()==field.as_dict()
    assert restored.electric_field_at(p)==pytest.approx(field.electric_field_at(p),rel=2e-13)


@pytest.mark.parametrize("bad",[
    dict(charge_c=0),dict(charge_c=True),dict(charge_c=float("nan")),
    dict(radius_m=0),dict(radius_m=-.1),dict(radius_m=True),
    dict(center_m=(0.,float("inf"),0.)),dict(axis=(0.,0.,0.)),
    dict(axis=(1.,float("nan"),3.)),
    dict(center_m=(1e12,0.,0.),radius_m=.1),
])
def test_n2_malformed_or_unresolved_source_fails(bad):
    with pytest.raises((ValueError,UnsupportedProblem)):
        ring(**bad)


def test_n1_exact_wire_pole_and_ulp_near_wire_are_refused():
    s=ring()
    for p in ((.2,0,0),(.2+1e-15,0,0),(.2,1e-16,0)):
        assert s.clearance_at(p)<1e-12
        with pytest.raises(SingularField,match="ring wire"):
            s.potential_at(p)
        with pytest.raises(SingularField,match="ring wire"):
            s.electric_field_at(p)
    near=(.2,0,.001)
    assert s.potential_at(near)>0
    assert all(abs(x)<1e10 for x in s.electric_field_at(near))
    assert s.clearance_at(near)==pytest.approx(.001,rel=1e-12)


def test_n4_static_source_does_not_implicitly_enable_particle_dynamics():
    s=ring()
    with pytest.raises(UnsupportedProblem,match="static E/V only"):
        TestChargeMotionSpec(PrescribedField((s,)),PointCharge(1e-9,(0,0,.1)),
                             1e-6,(.1,0.,0.),.1,.01)
    from conductor_sim.dipoles import RigidDipole,DipoleModel
    from conductor_sim.structural_scenarios import NamedSource
    from conductor_sim.dipole_motion import DipoleMotionSpec
    dm=DipoleModel(RigidDipole(1e-9,.01,pivot_m=(0,0,0)),(
        NamedSource("ring:new","Full static ring",s),),
        "test:dipole","Teacher hypothetical static dipole ring.")
    with pytest.raises(UnsupportedProblem,match="static ring"):
        DipoleMotionSpec(model=dm,rotation_axis=(0.,0.,1.),
            inertia_kg_m2=1e-6,initial_angle_rad=0.,initial_angular_velocity_rad_s=0.,
            duration_s=.1,sample_interval_s=.01)


@pytest.fixture(scope="module")
def workbook():
    from examples.m27a_ring_field import make_case
    return make_case("workbook-two-rings")


def test_p4_two_world_x_rings_upward_E_and_no_fictitious_axial_motion(workbook):
    record=workbook
    sources=record["field"]["sources"]
    assert [s["kind"] for s in sources]==["UniformRingField","UniformRingField","UniformElectricField"]
    assert sources[0]["axis"]==[1.,0.,0.] and sources[1]["axis"]==[1.,0.,0.]
    assert sources[0]["charge_c"]==10e-6 and sources[1]["charge_c"]==-20e-6
    assert sources[0]["radius_m"]==sources[1]["radius_m"]==.3
    assert sources[1]["center_m"]==[.4,0.,0.]
    assert sources[2]["field_v_m"]==[0.,100000.,0.]
    assert record["plane"]["u"]==[1.,0.,0.]
    assert record["plane"]["v"]==[0.,0.,1.]
    assert record["motion_qualified"] is False and record["time_evolution"] is False
    cells=record["grid"]["cells"]
    assert sum(c["region"]=="source-exclusion" for c in cells)>0
    for c in cells:
        if c["region"]=="physical-observation":
            assert c["electric_field_v_m"][1]==pytest.approx(100000.,rel=2e-15,abs=1e-6)
            assert c["field_normal_v_m"]==pytest.approx(-100000.,rel=2e-15,abs=1e-6)
    assert validate_ring_field_slice(record)
    html=render_ring_field_slice(record,title="Two charged rings and transverse E")
    assert html.count('class="ring-source-projection"')==2
    assert 'source-exclusion' in html and 'No conductor induction' in html
    assert 'type="application/json"' in html
    assert '<script>' not in html and 'elapsed-time' not in html


def test_n5_2d_map_retains_offplane_real_field_and_masks_actual_wire():
    s=ring(radius_m=.12)
    field=PrescribedField((s,UniformElectricField((0,0,5.))))
    plane=SliceSpec(origin_m=(0.,0.,0.),u=(1.,0.,0.),v=(0.,1.,0.),
                    x_range_m=(-.2,.2),y_range_m=(-.2,.2),resolution=21)
    record=sample_ring_field_slice(field,plane,
             source_annotations=(("ring:R","Ring R"),("field:normal","Normal uniform E")),
             exclusion_width_m=.006)
    assert validate_ring_field_slice(record)
    n=record["grid"]["rows"];cell=record["grid"]["cells"][10*n+16]
    assert cell["region"]=="source-exclusion" and cell["potential_v"] is None
    unmasked=next(c for c in record["grid"]["cells"] if c["region"]=="physical-observation")
    assert unmasked["field_normal_v_m"]==pytest.approx(5.,abs=1e-12)
    # Historical M21b certifier is intentionally unchanged; new M27a
    # observation is opt-in rather than a hidden migration of old maps.
    from conductor_sim.jee_potential_explorer import sample_fixed_potential_map
    with pytest.raises(UnsupportedProblem):
        sample_fixed_potential_map(field,plane,source_annotations=(("ring:R","Ring R"),("field:normal","Normal uniform E")),
                                   source_mask_m=.006)


@pytest.mark.parametrize("edit",[
    lambda r:r["grid"]["cells"][0].__setitem__("potential_v",999999.),
    lambda r:r["grid"]["cells"][0].__setitem__("field_normal_v_m",999999.),
    lambda r:r["ring_projections"][0].__setitem__("source_id","forged:id"),
    lambda r:r.__setitem__("motion_qualified",True),
    lambda r:r["grid"]["cells"][0].__setitem__("world_m",[0.,0.,0.]),
    lambda r:r.__setitem__("time_s",100.),
])
def test_n6_forged_static_records_rejected(workbook,edit):
    tampered=deepcopy(workbook)
    edit(tampered)
    with pytest.raises((ValueError,TypeError,KeyError)):
        validate_ring_field_slice(tampered)


def test_n5_masked_cell_cannot_be_replaced_with_fabricated_zero(workbook):
    tampered=deepcopy(workbook)
    masked=next(c for c in tampered["grid"]["cells"] if c["region"]=="source-exclusion")
    masked["potential_v"]=0.
    masked["electric_field_v_m"]=[0.,0.,0.]
    with pytest.raises(ValueError,match="source mask"):
        validate_ring_field_slice(tampered)
