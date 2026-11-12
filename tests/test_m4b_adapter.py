"""M4b purely mocked FEMM compiler/result contract for nested shells."""
from math import pi
import pytest

from conductor_sim import ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell
from conductor_sim.femm5 import Femm5Backend, _SOURCE_CORE
from conductor_sim.solver import SingularField, UnsupportedProblem
from test_femm5_adapter import FakeClient, FakeSession, calls


class NestedFakeClient(FakeClient):
    def eo_getconductorproperties(self,name):
        self.calls.append(('eo_getconductorproperties',(name,)))
        return {'inner':(30.0,0.0),'outer':(0.0,-1e-9)}.get(name,(150.0,1e-9))


def p(first=FixedNetCharge(0),second=Grounded(),reverse=False,charges=(PointCharge(1e-9,(0,0,.035)),)):
    shells=(SphericalShell(.1,.12,first,name='inner'),
            SphericalShell(.2,.22,second,name='outer'))
    return ElectrostaticProblem(charges,tuple(reversed(shells)) if reverse else shells)


def backend():
    client=NestedFakeClient()
    session=FakeSession(client)
    return Femm5Backend(session_factory=lambda:session),client,session


def test_nested_shells_use_independent_physical_conductors_and_source():
    b,c,s=backend()
    obj=p()
    assert b.supports(obj)
    with b.solve(obj) as sol:
        assert [x.name for x in sol.radial_shells]==['inner','outer']
        assert sol.shell.name=='inner'
        assert len(sol.conductor_results)==2
        assert sol.conductor_by_name('inner').net_charge_c==0
        assert sol.conductor_by_name('inner').inner_surface_charge_c==pytest.approx(-1e-9)
        assert sol.conductor_by_name('inner').outer_surface_charge_c==pytest.approx(1e-9)
        assert sol.conductor_by_name('outer').net_charge_c==pytest.approx(-1e-9)
        assert sol.conductor_by_name('outer').inner_surface_charge_c==pytest.approx(-1e-9)
        assert sol.conductor_by_name('outer').outer_surface_charge_c==pytest.approx(0)
        assert len(sol.numerical_sources)==1
        for r in (.1,.12,.2,.22):
            with pytest.raises(SingularField):
                sol.electric_field_at((r,0,0))
    assert s.opens==s.closes==1
    assert calls(c,'ei_addconductorprop')==[
        ('inner',0,0,0),('outer',0,0,1),(_SOURCE_CORE,0,1e-9,0)]
    assert len(calls(c,'ei_addarc'))==12  # 4 shell surfaces, outer domain, 1 core
    assert len(calls(c,'ei_addsegment'))==11
    assert len(calls(c,'ei_addblocklabel'))==6 # 1 core + cavity + 2 metals + gap + exterior
    arcs=calls(c,'ei_setarcsegmentprop')
    assert sum(entry[4]=='inner' for entry in arcs)==4
    assert sum(entry[4]=='outer' for entry in arcs)==4
    assert sum(entry[4]==_SOURCE_CORE for entry in arcs)==2
    assert sum(entry[1]=='open_robin' for entry in arcs)==2


def test_reversed_author_order_preserves_authored_result_order():
    b,c,_=backend()
    obj=p(reverse=True)
    with b.solve(obj) as sol:
        assert [x.name for x in sol.conductor_results]==['outer','inner']
        assert [x.name for x in sol.radial_shells]==['inner','outer']
        assert sol.conductor_by_name('outer').net_charge_c==pytest.approx(-1e-9)
    assert calls(c,'ei_addconductorprop')[:2]==[('inner',0,0,0),('outer',0,0,1)]


def test_surface_sigma_requires_name_with_multiple_shells():
    b,c,_=backend()
    with b.solve(p()) as sol:
        with pytest.raises(UnsupportedProblem,match='conductor_name'):
            sol.surface_charge_density('inner',pi/2)
        eps=8.8541878128e-12
        assert sol.surface_charge_density('inner',pi/2,conductor_name='inner')==pytest.approx(-4*eps)
        assert sol.surface_charge_density('inner',pi/2,conductor_name='outer')==pytest.approx(-4*eps)
        assert sol.surface_charge_density('outer',pi/2,conductor_name='outer')==pytest.approx(4*eps)
        with pytest.raises(ValueError,match='unknown physical conductor'):
            sol.surface_charge_density('outer',.4,conductor_name='bogus')


def test_two_shells_and_two_sources_both_compose():
    obj=p(charges=(PointCharge(1e-9,(0,0,.035)),PointCharge(.4e-9,(0,0,-.025))))
    b,c,_=backend()
    with b.solve(obj) as sol:
        assert len(sol.numerical_sources)==2
        assert len(sol.conductor_results)==2
    assert len(calls(c,'ei_addarc'))==14
    assert len(calls(c,'ei_addblocklabel'))==7
    assert len(calls(c,'ei_addsegment'))==13


def test_nonconcentric_touching_and_charge_in_gap_rejected_without_session():
    b,c,s=backend()
    bad=[
        ElectrostaticProblem(conductors=(
            SphericalShell(.1,.2,Grounded(),name='inner'),
            SphericalShell(.2,.3,Grounded(),name='outer'))),
        ElectrostaticProblem(conductors=(
            SphericalShell(.1,.12,Grounded(),name='inner'),
            SphericalShell(.2,.22,Grounded(),center_m=(0,0,.01),name='outer'))),
        p(charges=(PointCharge(1e-9,(0,0,.15)),)),
        p(charges=(PointCharge(1e-9,(0,0,.035)),PointCharge(1e-9,(.02,0,0)))),
    ]
    for obj in bad:
        assert not b.supports(obj)
        with pytest.raises(UnsupportedProblem):
            b.solve(obj)
    assert s.opens==0


def test_empty_nested_shells_keep_correct_region_labels():
    b,c,_=backend()
    with b.solve(p(charges=())) as sol:
        assert not sol.numerical_sources
        assert len(sol.conductor_results)==2
    assert len(calls(c,'ei_addarc'))==10
    assert len(calls(c,'ei_addblocklabel'))==5 # cavity, metal1, gap, metal2, exterior


def test_limit_number_of_shells():
    obj=ElectrostaticProblem(conductors=tuple(
        SphericalShell(.1+i*.05,.12+i*.05,FixedNetCharge(0),name=f's{i}')
        for i in range(9)))
    assert not Femm5Backend().supports(obj)
