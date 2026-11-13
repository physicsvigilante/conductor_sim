"""M14a: independently check wire equal-potential, charge and energy identities."""
from dataclasses import replace
from math import pi, fsum
import json

import pytest

from conductor_sim import (ConnectionPlan, ConnectedShellBackend, ElectricalLink,
    ElectrostaticProblem, FixedNetCharge, Grounded, NestedShellBackend,
    PointCharge, SphericalShell)
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.visualization import write_visualization

K=1/(4*pi*8.8541878128e-12)
Q=1e-9


def shells(*nets):
    radii=((.08,.1),(.2,.22),(.33,.35),(.46,.49))
    return tuple(SphericalShell(radii[i][0],radii[i][1],constraint,name=f's{i}')
                 for i,constraint in enumerate(nets))


def connect(problem, *links):
    return ConnectedShellBackend().solve(ConnectionPlan(problem, tuple(links)))


def test_two_floating_shells_equal_potential_and_excess_charge_moves_to_outermost():
    p=ElectrostaticProblem(conductors=shells(FixedNetCharge(Q), FixedNetCharge(0)))
    result=connect(p,ElectricalLink(('s0','s1')))
    before,after=result.before,result.after
    assert p.conductors[0].constraint.charge_c==Q  # immutable authored initial model
    assert result.plan.initial is p
    assert result.resolved_problem.charges is p.charges
    assert [c.name for c in result.resolved_problem.conductors]==['s0','s1']
    assert after.conductor_by_name('s0').net_charge_c==pytest.approx(0,abs=1e-23)
    assert after.conductor_by_name('s1').net_charge_c==pytest.approx(Q)
    assert after.conductor_by_name('s0').potential_v==pytest.approx(K*Q/.22)
    assert after.conductor_by_name('s1').potential_v==pytest.approx(K*Q/.22)
    assert after.potential_at((.15,0,0))==pytest.approx(K*Q/.22)
    assert after.electric_field_at((.15,0,0))==pytest.approx((0.,0.,0.),abs=1e-15)
    expected_heat=.5*K*Q**2*(1/.1-1/.2)
    assert result.released_field_energy_j==pytest.approx(expected_heat)
    assert result.radial_field_energy_after_j==pytest.approx(.5*K*Q**2/.22)
    assert result.radial_field_energy_before_j==pytest.approx(.5*K*Q**2*(1/.1-1/.2+1/.22))
    assert [t.received_c for t in result.transfers]==pytest.approx((-Q,Q))
    # View uses two existing model equilibria, not a browser-time wire solver.
    bundle=result.visualization(resolution=11,line_steps=12,seeds=[],profile_samples=9)
    assert bundle['schema']=='conductor_sim.visualization.v1'
    assert len(bundle['frames'])==2
    assert len(bundle['frames'][0]['geometry'])==2
    assert len(bundle['frames'][1]['profiles'])==4
    assert bundle['connection']['released_field_energy_j']==pytest.approx(expected_heat)
    assert 'not a transient' in bundle['connection']['playback_meaning']
    json.dumps(bundle,allow_nan=False)


def test_charge_inside_cavity_connection_removes_gap_flux_but_not_cavity_field():
    q=2*Q
    charge=PointCharge(q,(0,.02,.035))
    p=ElectrostaticProblem((charge,),shells(FixedNetCharge(Q),FixedNetCharge(-.5*Q)))
    r=connect(p,ElectricalLink(('s1','s0')))
    a,b=r.after.conductor_by_name('s0'),r.after.conductor_by_name('s1')
    assert a.net_charge_c==pytest.approx(-q)
    assert b.net_charge_c==pytest.approx(2.5*Q)
    assert a.inner_surface_charge_c==pytest.approx(-q)
    assert a.outer_surface_charge_c==pytest.approx(0,abs=1e-22)
    assert b.inner_surface_charge_c==pytest.approx(0,abs=1e-22)
    assert b.outer_surface_charge_c==pytest.approx(2.5*Q)
    assert r.after.electric_field_at((.15,0,0))==pytest.approx((0.,0.,0.),abs=1e-12)
    for point in ((.04,0,0),(.025,.04,.035),(.0,-.03,.02)):
        assert r.before.electric_field_at(point)==pytest.approx(r.after.electric_field_at(point))
        # The cavity potential can change only by the new shell potential offset.
        voltage_delta=(r.after.conductor_by_name('s0').potential_v-
                       r.before.conductor_by_name('s0').potential_v)
        assert r.after.potential_at(point)-r.before.potential_at(point)==pytest.approx(voltage_delta)
    assert r.released_field_energy_j==pytest.approx(.5*K*(q+Q)**2*(1/.1-1/.2))
    assert r.resolved_problem.permittivity_f_per_m==p.permittivity_f_per_m


def test_wire_to_ground_is_not_charge_conserving_and_both_shells_are_grounded():
    p=ElectrostaticProblem(conductors=shells(FixedNetCharge(Q),Grounded()))
    result=connect(p,ElectricalLink(('s0','s1'),'grounded'))
    assert all(isinstance(s.constraint,Grounded) for s in result.resolved_problem.conductors)
    assert all(abs(result.after.conductor_by_name(s.name).potential_v)<1e-8
               for s in p.conductors)
    assert all(abs(result.after.conductor_by_name(s.name).net_charge_c)<1e-21
               for s in p.conductors)
    assert result.before.conductor_by_name('s1').net_charge_c==pytest.approx(-Q)
    assert result.after.potential_at((.28,0,0))==pytest.approx(0,abs=1e-10)
    assert result.released_field_energy_j==pytest.approx(.5*K*Q**2*(1/.1-1/.2))
    assert sum(t.received_c for t in result.transfers)==pytest.approx(0,abs=1e-21)
    # In this example, grounding the outer shell already removed the exterior charge;
    # the connecting wire eliminates the field in the gap and changes inner charge.
    assert result.transfers[0].received_c==pytest.approx(-Q)
    assert result.transfers[1].received_c==pytest.approx(+Q)


def test_nonadjacent_wire_with_a_floating_middle_shell_uses_general_matrix():
    qm=.4*Q
    p=ElectrostaticProblem(conductors=shells(FixedNetCharge(1.3*Q),FixedNetCharge(qm),FixedNetCharge(-.2*Q)))
    result=connect(p,ElectricalLink(('s0','s2')))
    g0=1/.1-1/.2
    g1=1/.22-1/.33
    q_inner=-qm*g1/(g0+g1)
    q_outer=1.1*Q-q_inner
    assert result.after.conductor_by_name('s0').net_charge_c==pytest.approx(q_inner)
    assert result.after.conductor_by_name('s1').net_charge_c==pytest.approx(qm)
    assert result.after.conductor_by_name('s2').net_charge_c==pytest.approx(q_outer)
    assert result.after.conductor_by_name('s0').potential_v==pytest.approx(
        result.after.conductor_by_name('s2').potential_v,abs=1e-8)
    assert result.after.electric_field_at((.15,0,0))[0]*result.after.electric_field_at((.27,0,0))[0]<0
    assert result.released_field_energy_j>0
    assert result.after.conductor_by_name('s1').potential_v!=pytest.approx(result.after.conductor_by_name('s0').potential_v)
    # Independently verify variational extremum: perturbing the group charge
    # partition either way raises field energy at conserved group total.
    for change in (-.1*Q,.1*Q):
        trial=ElectrostaticProblem(conductors=shells(FixedNetCharge(q_inner+change),
             FixedNetCharge(qm),FixedNetCharge(q_outer-change)))
        static=NestedShellBackend().solve(trial)
        from conductor_sim.connections import _field_energy
        assert _field_energy(static)>result.radial_field_energy_after_j


def test_two_separate_connections_in_four_shells_preserve_group_charge_totals():
    p=ElectrostaticProblem(conductors=shells(FixedNetCharge(Q),FixedNetCharge(.2*Q),
        FixedNetCharge(-.5*Q),FixedNetCharge(.7*Q)))
    links=(ElectricalLink(('s2','s3')),ElectricalLink(('s0','s1')))
    r=connect(p,*links)
    for names,total in ((('s0','s1'),1.2*Q),(('s2','s3'),.2*Q)):
        assert sum(r.after.conductor_by_name(n).net_charge_c for n in names)==pytest.approx(total)
        assert r.after.conductor_by_name(names[0]).potential_v==pytest.approx(
            r.after.conductor_by_name(names[1]).potential_v)
    assert r.released_field_energy_j>0
    assert sum(x.received_c for x in r.transfers)==pytest.approx(0,abs=1e-21)


def test_reverse_input_shell_order_and_link_member_order_do_not_change_physics():
    p=ElectrostaticProblem(conductors=shells(FixedNetCharge(Q),FixedNetCharge(-.3*Q),FixedNetCharge(.1*Q)))
    first=connect(p,ElectricalLink(('s0','s2')))
    reversed_problem=ElectrostaticProblem(p.charges,tuple(reversed(p.conductors)))
    second=connect(reversed_problem,ElectricalLink(('s2','s0')))
    assert [x.name for x in second.transfers]==['s2','s1','s0']
    for name in ('s0','s1','s2'):
        assert first.after.conductor_by_name(name)==pytest.approx(second.after.conductor_by_name(name))
    assert first.released_field_energy_j==pytest.approx(second.released_field_energy_j)


def test_translation_rotated_point_source_and_charge_scaling():
    charges=(PointCharge(Q,(.014,.012,.016)),PointCharge(-.45*Q,(-.016,0,.003)))
    p=ElectrostaticProblem(charges,shells(FixedNetCharge(.7*Q),FixedNetCharge(.2*Q)))
    r=connect(p,ElectricalLink(('s0','s1')))
    center=(3.,-.7,.9)
    rotated=lambda x: (center[0]+x[2],center[1]+x[1],center[2]-x[0])
    cp=ElectrostaticProblem(tuple(PointCharge(q.charge_c,rotated(q.position_m)) for q in charges),
        tuple(replace(s,center_m=center) for s in p.conductors))
    other=connect(cp,ElectricalLink(('s0','s1')))
    assert other.released_field_energy_j==pytest.approx(r.released_field_energy_j,rel=1e-11)
    for name in ('s0','s1'):
        assert other.after.conductor_by_name(name).net_charge_c==pytest.approx(r.after.conductor_by_name(name).net_charge_c)
    assert other.after.electric_field_at(rotated((.035,.01,.01)))==pytest.approx(
        (r.after.electric_field_at((.035,.01,.01))[2],
         r.after.electric_field_at((.035,.01,.01))[1],
        -r.after.electric_field_at((.035,.01,.01))[0]),rel=1e-10)
    double=ElectrostaticProblem(tuple(PointCharge(2*c.charge_c,c.position_m) for c in charges),
        tuple(replace(s,constraint=FixedNetCharge(2*s.constraint.charge_c)) for s in p.conductors))
    scaled=connect(double,ElectricalLink(('s0','s1')))
    assert scaled.released_field_energy_j==pytest.approx(4*r.released_field_energy_j)


def test_reject_unsupported_networks_before_solver_or_wrong_physical_model():
    b=ConnectedShellBackend()
    p=ElectrostaticProblem(conductors=shells(FixedNetCharge(Q),FixedNetCharge(0)))
    bad=[
        ConnectionPlan(p,(ElectricalLink(('s0','missing')),)),
        ConnectionPlan(ElectrostaticProblem(conductors=shells(Grounded(),FixedNetCharge(Q))),
                       (ElectricalLink(('s0','s1')),)),
        ConnectionPlan(ElectrostaticProblem(conductors=(p.conductors[0],)),
                       (ElectricalLink(('s0','s1')),)),
        ConnectionPlan(p,(ElectricalLink(('s0','s1')),ElectricalLink(('s1','s0')))),
        ConnectionPlan(ElectrostaticProblem(conductors=(p.conductors[0],
                       replace(p.conductors[1],center_m=(.01,0,0)))),
                       (ElectricalLink(('s0','s1')),)),
        ConnectionPlan(ElectrostaticProblem(charges=(PointCharge(Q,(.15,0,0)),),
                       conductors=p.conductors), (ElectricalLink(('s0','s1')),)),
    ]
    for plan in bad:
        assert not b.supports(plan)
        with pytest.raises(UnsupportedProblem):
            b.solve(plan)
    for args in ((('s0',),'floating'),(('s0','s0'),'floating'),(('s0','s1'),'invalid')):
        with pytest.raises(ValueError):
            ElectricalLink(*args)
    with pytest.raises(ValueError):
        ConnectionPlan(p,())
    with pytest.raises(TypeError):
        ConnectionPlan(None,(ElectricalLink(('s0','s1')),))
    with pytest.raises(UnsupportedProblem):
        b.solve(p)


def test_overlapping_connections_do_not_change_initial_model_even_when_error():
    p=ElectrostaticProblem(conductors=shells(FixedNetCharge(Q),FixedNetCharge(0),FixedNetCharge(0)))
    plan=ConnectionPlan(p,(ElectricalLink(('s0','s1')),ElectricalLink(('s1','s2'))))
    with pytest.raises(UnsupportedProblem):
        ConnectedShellBackend().solve(plan)
    assert p.conductors[0].constraint.charge_c==Q


def test_grounded_link_charge_inside_cavity_and_ungrounded_middle_constraint():
    p=ElectrostaticProblem((PointCharge(Q,(0,0,.035)),),shells(FixedNetCharge(.3*Q),FixedNetCharge(-.2*Q),Grounded()))
    plan=ConnectionPlan(p,(ElectricalLink(('s0','s2'),'grounded'),))
    r=ConnectedShellBackend().solve(plan)
    assert r.after.conductor_by_name('s0').potential_v==pytest.approx(0,abs=1e-8)
    assert r.after.conductor_by_name('s2').potential_v==pytest.approx(0,abs=1e-8)
    assert r.after.conductor_by_name('s1').net_charge_c==pytest.approx(-.2*Q)
    assert r.released_field_energy_j>=0


def test_incompatible_conductor_shape_is_explicitly_unsupported():
    from conductor_sim import SolidSphere, ConductingPlane
    b=ConnectedShellBackend()
    for conductors in (
        (SolidSphere(.1,FixedNetCharge(Q),name='s0'),
         SphericalShell(.2,.22,FixedNetCharge(0),name='s1')),
        (ConductingPlane(name='s0'),
         SphericalShell(.2,.22,FixedNetCharge(0),name='s1')),
    ):
        plan=ConnectionPlan(ElectrostaticProblem(conductors=conductors),
                             (ElectricalLink(('s0','s1')),))
        assert not b.supports(plan)
        with pytest.raises(UnsupportedProblem,match='SphericalShell'):
            b.solve(plan)


def test_energy_derivative_matches_potential_and_connection_minimizes_energy():
    from conductor_sim.connections import _field_energy
    q0=.6*Q
    baseline=(.7*Q,-.3*Q,.4*Q)
    p=ElectrostaticProblem((PointCharge(q0,(0,0,.03)),),
        shells(*(FixedNetCharge(x) for x in baseline)))
    before=NestedShellBackend().solve(p)
    eps=1e-13
    for i in range(3):
        qlo=list(baseline);qlo[i]-=eps
        qhi=list(baseline);qhi[i]+=eps
        at=lambda charges: ElectrostaticProblem(p.charges,shells(*(FixedNetCharge(x) for x in charges)))
        deriv=(_field_energy(NestedShellBackend().solve(at(qhi)))-
               _field_energy(NestedShellBackend().solve(at(qlo))))/(2*eps)
        assert deriv==pytest.approx(before.conductor_by_name(f's{i}').potential_v,rel=1e-8)
    connected=connect(p,ElectricalLink(('s0','s1')))
    assert connected.released_field_energy_j>=0
    # Equal voltage is exactly the variational stationarity under fixed
    # total charge, even with the third shell independently isolated.
    assert connected.after.conductor_by_name('s0').potential_v==pytest.approx(
        connected.after.conductor_by_name('s1').potential_v,rel=1e-9)


def test_closed_form_single_cavity_source_and_no_net_wire_supply():
    # A linked floating pair conserves only the shells' initial combined Q;
    # an off-axis point charge in the cavity is not a member of the wire.
    q=-.6*Q
    charges=(PointCharge(q,(.015,0,.02)),)
    p=ElectrostaticProblem(charges,shells(FixedNetCharge(0),FixedNetCharge(0)))
    r=connect(p,ElectricalLink(('s0','s1')))
    assert r.after.conductor_by_name('s0').net_charge_c==pytest.approx(-q)
    assert r.after.conductor_by_name('s1').net_charge_c==pytest.approx(q)
    assert fsum(t.received_c for t in r.transfers)==pytest.approx(0,abs=1e-22)
    assert r.after.conductor_by_name('s1').outer_surface_charge_c==pytest.approx(q)
    assert r.released_field_energy_j==pytest.approx(.5*K*q*q*(1/.1-1/.2))


def test_pdf_page14_numerical_q7_exact_outer_potential_change_is_zero():
    from examples.connected_shells import scenario
    p=scenario('pdf-p14-q7')
    r=ConnectedShellBackend().solve(p)
    q=1e-9
    assert r.before.conductor_by_name('outer').potential_v==pytest.approx(K*3*q/.20)
    assert r.after.conductor_by_name('outer').potential_v==pytest.approx(K*3*q/.20)
    assert r.after.conductor_by_name('outer').potential_v-r.before.conductor_by_name('outer').potential_v==pytest.approx(0.,abs=1e-11)
    assert r.after.conductor_by_name('inner').net_charge_c==pytest.approx(0.,abs=1e-21)
    assert r.after.conductor_by_name('outer').net_charge_c==pytest.approx(3*q)
    assert r.released_field_energy_j==pytest.approx(K*q*q/2*(1/.10-1/.195))


def test_all_examples_reuse_exact_same_components_and_are_physically_consistent(tmp_path):
    from examples.connected_shells import scenario
    for case in ('pdf-p14-q7','ground-link','cavity-charge','three-shell'):
        plan=scenario(case)
        assert ConnectedShellBackend().supports(plan)
        resolved=ConnectedShellBackend().solve(plan)
        bundle=resolved.visualization(resolution=9,line_steps=12,seeds=[],profile_samples=9)
        assert len(bundle['frames'])==2
        assert len(bundle['connection']['transfers'])==len(plan.initial.conductors)
        assert 'not a current transient' in bundle['frames'][1]['notes'][-2]
        assert all(abs(resolved.after.conductor_by_name(name).potential_v-resolved.after.conductor_by_name(plan.links[0].members[0]).potential_v)<1e-8 for name in plan.links[0].members)
        path=tmp_path/(case+'.html')
        write_visualization(bundle,path)
        assert path.is_file()
        assert 'id="scene"' in path.read_text(encoding='utf-8')
