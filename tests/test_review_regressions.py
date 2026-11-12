"""Reproductions from the post-M9 correctness review; no mocked accuracy claims."""
from itertools import permutations
from math import isfinite, pi
from pathlib import Path

import pytest

from conductor_sim import (
    ConcentricShellBackend, ElectrostaticProblem, ExternalGroundedSphereBackend,
    ExternalIsolatedSphereBackend, ExternalSphereBackend, Femm5Backend,
    Femm5ExecutionError, Femm5ExternalSphereBackend, Femm5Settings, FixedNetCharge,
    Grounded, ImageChargeShellBackend, NestedShellBackend, PointCharge,
    SolidSphere, SphericalShell, UnsupportedProblem,
)
from conductor_sim.visualization import SliceSpec, sample_sweep
from test_femm5_adapter import FakeClient, FakeSession, calls


def cavity_problem(name='metal'):
    return ElectrostaticProblem((PointCharge(1e-9),),
                                (SphericalShell(.1, .12, Grounded(), name=name),))


def external_problem(name='metal'):
    return ElectrostaticProblem((PointCharge(1e-9, (0, 0, .27)),),
                                (SolidSphere(.12, Grounded(), name=name),))


@pytest.mark.parametrize('backend_cls,factory', [
    (Femm5Backend, cavity_problem), (Femm5ExternalSphereBackend, external_problem),
])
def test_femm_reserved_no_conductor_name_rejected_before_session(backend_cls, factory):
    session = FakeSession(FakeClient())
    backend = backend_cls(session_factory=lambda: session)
    problem = factory('<None>')
    assert not backend.supports(problem)
    with pytest.raises(UnsupportedProblem):
        backend.solve(problem)
    assert session.opens == 0


@pytest.mark.parametrize('key,value', [
    ('arc_degrees', 10.1), ('arc_degrees', 15),
    ('solver_precision', 1e-7), ('solver_precision', 1e-17),
])
def test_requested_settings_must_not_be_silently_clamped_by_pinned_femm(key, value):
    with pytest.raises(ValueError):
        Femm5Settings(**{key: value})


@pytest.mark.parametrize('precision', [1e-16, 1e-8])
def test_actual_femm_precision_endpoints_remain_accepted(precision):
    assert Femm5Settings(arc_degrees=10, solver_precision=precision).solver_precision == precision


@pytest.mark.parametrize('grounded', [True, False])
def test_exact_exterior_density_finite_for_distinct_near_contact_source(grounded):
    a = .1
    d = a + 1e-10
    q = 1e-9
    constraint = Grounded() if grounded else FixedNetCharge(0)
    problem = ElectrostaticProblem((PointCharge(q, (0, 0, d)),), (SolidSphere(a, constraint),))
    solution = ExternalSphereBackend().solve(problem)
    # Evaluate the pole directly from the physical gap, without subtracting
    # nearly equal squared lengths, and add the isolated compensator.
    expected = -q * (d + a) / (4*pi*a*(d-a)**2)
    if not grounded:
        expected += (q*a/d) / (4*pi*a*a)
    assert isfinite(solution.surface_charge_density('outer', 0))
    assert solution.surface_charge_density('outer', 0) == pytest.approx(expected, rel=2e-13)
    assert solution.surface_charge_density_at('outer', (0, 0, a)) == pytest.approx(expected, rel=2e-13)


def test_exact_interior_density_finite_for_distinct_near_wall_source():
    a = .1
    d = a - 1e-10
    q = 1e-9
    problem = ElectrostaticProblem((PointCharge(q, (0, 0, d)),),
                                   (SphericalShell(a, .12, Grounded()),))
    solution = ImageChargeShellBackend().solve(problem)
    expected = -q * (a+d) / (4*pi*a*(a-d)**2)
    assert isfinite(solution.surface_charge_density('inner', 0))
    assert solution.surface_charge_density('inner', 0) == pytest.approx(expected, rel=2e-13)


class CloseFailure(FakeSession):
    def close(self):
        super().close()
        raise RuntimeError('synthetic cleanup fault')


@pytest.mark.parametrize('backend_cls,factory', [
    (Femm5Backend, cavity_problem), (Femm5ExternalSphereBackend, external_problem),
])
def test_solver_failure_survives_secondary_close_failure(backend_cls, factory):
    client = FakeClient(fail_on='ei_analyze')
    session = CloseFailure(client)
    with pytest.raises(Femm5ExecutionError, match='synthetic FEMM fault') as caught:
        backend_cls(session_factory=lambda: session).solve(factory())
    assert 'synthetic cleanup fault' in repr(getattr(caught.value, 'cleanup_errors', ()))
    assert session.closes == 1
    assert not Path(calls(client, 'ei_saveas')[0][0]).parent.exists()


@pytest.mark.parametrize('backend_cls,factory', [
    (Femm5Backend, cavity_problem), (Femm5ExternalSphereBackend, external_problem),
])
def test_body_exception_survives_context_manager_close_failure(backend_cls, factory):
    client = FakeClient()
    session = CloseFailure(client)
    error = ValueError('primary body fault')
    with pytest.raises(ValueError, match='primary body fault') as caught:
        with backend_cls(session_factory=lambda: session).solve(factory()):
            raise error
    assert caught.value is error
    assert 'synthetic cleanup fault' in repr(getattr(error, 'cleanup_errors', ()))
    assert session.closes == 1
    assert not Path(calls(client, 'ei_saveas')[0][0]).parent.exists()


@pytest.mark.parametrize('backend_cls,factory', [
    (Femm5Backend, cavity_problem), (Femm5ExternalSphereBackend, external_problem),
])
def test_cleanup_failure_after_success_is_not_silently_ignored(backend_cls, factory):
    client = FakeClient()
    session = CloseFailure(client)
    with pytest.raises(RuntimeError, match='synthetic cleanup fault'):
        with backend_cls(session_factory=lambda: session).solve(factory()):
            pass
    assert not Path(calls(client, 'ei_saveas')[0][0]).parent.exists()


@pytest.mark.parametrize('point', [None, 1, 'abc', (0, 0), (0, 0, 'z'), (0, 0, float('nan'))])
def test_shell_femm_samples_validate_the_same_coordinate_contract(point):
    session = FakeSession(FakeClient())
    with Femm5Backend(session_factory=lambda: session).solve(cavity_problem()) as solution:
        for query in (solution.potential_at, solution.electric_field_at):
            with pytest.raises(ValueError):
                query(point)


def test_one_shot_field_line_seeds_reused_for_every_stored_frame():
    problem = cavity_problem()
    bundle = sample_sweep([('first', problem), ('same', problem)], NestedShellBackend(),
                          plane=SliceSpec(resolution=9), seeds=iter([((.05, 0), 1)]), line_steps=12)
    first, second = bundle['frames']
    assert first['field_lines']
    assert second['field_lines'] == first['field_lines']
    assert second['line_termination_counts'] == first['line_termination_counts']


def test_seed_budget_is_checked_before_opening_any_solver():
    class MustNotSolve:
        def solve(self, problem):
            raise AssertionError('invalid export opened a solver')
    with pytest.raises(ValueError, match='96'):
        sample_sweep([('bad', cavity_problem())], MustNotSolve(),
                     seeds=(((.05, 0), 1) for _ in range(97)))


def test_default_external_view_includes_its_sources_with_margin():
    problem = external_problem()
    bundle = sample_sweep([('exterior', problem)], ExternalSphereBackend(), resolution=9, line_steps=12)
    frame = bundle['frames'][0]
    u, v, _ = frame['charges'][0]['slice_position_m']
    assert frame['plane']['x_range_m'][0] < u < frame['plane']['x_range_m'][1]
    assert frame['plane']['y_range_m'][0] < v < frame['plane']['y_range_m'][1]


def test_explicit_small_slice_not_resized_by_export():
    plane = SliceSpec(resolution=9, x_range_m=(-.1,.1), y_range_m=(-.1,.1))
    frame = sample_sweep([('exterior', external_problem())], ExternalSphereBackend(),
                         plane=plane, line_steps=12)['frames'][0]
    assert frame['plane']['y_range_m'] == (-.1,.1)


@pytest.mark.parametrize('axis', [(1, 0), (1, 0, 0, 0), None])
def test_explicit_malformed_meridian_axis_is_rejected(axis):
    with pytest.raises(ValueError):
        SliceSpec.meridian(cavity_problem(), axis=axis)


def test_center_charge_summation_is_order_independent_on_supported_python_versions():
    expected = 1e-26
    for values in permutations((1e-9, expected, -1e-9)):
        problem = ElectrostaticProblem(tuple(PointCharge(q) for q in values),
                                       (SphericalShell(.1, .12, Grounded()),))
        result = ConcentricShellBackend().solve(problem)
        assert result.charge_c == expected


def test_nested_enclosed_charge_does_not_drop_a_small_middle_shell_charge():
    problem = ElectrostaticProblem((PointCharge(1e-9),), (
        SphericalShell(.1,.12,FixedNetCharge(1e-26),name='inner'),
        SphericalShell(.2,.22,FixedNetCharge(-1e-9),name='outer')))
    solution = NestedShellBackend().solve(problem)
    assert solution.conductor_by_name('outer').outer_surface_charge_c == 1e-26


@pytest.mark.parametrize('kind', ['interior', 'exterior'])
def test_single_and_superposed_density_agree_at_close_but_distinct_source(kind):
    from conductor_sim import MultiImageShellBackend, ExternalMultiSphereBackend
    a, q = .1, 1e-9
    d = a + (1e-10 if kind == 'exterior' else -1e-10)
    metal = SolidSphere(a, Grounded()) if kind == 'exterior' else SphericalShell(a, .12, Grounded())
    point = (0., 0., d)
    two = ElectrostaticProblem((PointCharge(q/2, point), PointCharge(q/2, point)), (metal,))
    one = ElectrostaticProblem((PointCharge(q, point),), (metal,))
    if kind == 'exterior':
        exact = ExternalGroundedSphereBackend().solve(one).surface_charge_density('outer', 0)
        actual = ExternalMultiSphereBackend().solve(two).surface_charge_density_at('outer', (0, 0, a))
    else:
        exact = ImageChargeShellBackend().solve(one).surface_charge_density('inner', 0)
        actual = MultiImageShellBackend().solve(two).surface_charge_density_at('inner', (0, 0, a))
    assert actual == pytest.approx(exact, rel=2e-13)


@pytest.mark.parametrize('backend_cls,factory', [
    (Femm5Backend, cavity_problem), (Femm5ExternalSphereBackend, external_problem),
])
def test_cancelled_analysis_still_closes_session_and_workspace(backend_cls, factory):
    client = FakeClient()
    interrupt = KeyboardInterrupt('cancel solve')
    def cancel(*args):
        raise interrupt
    client.ei_analyze = cancel
    session = FakeSession(client)
    with pytest.raises(KeyboardInterrupt) as caught:
        backend_cls(session_factory=lambda: session).solve(factory())
    assert caught.value is interrupt
    assert session.closes == 1
    assert not Path(calls(client, 'ei_saveas')[0][0]).parent.exists()


def test_all_cleanup_actions_run_and_secondary_failures_remain_observable():
    from conductor_sim._resources import cleanup_resources
    attempted = []
    def fail(name):
        def run():
            attempted.append(name)
            raise RuntimeError(name)
        return run
    error = ValueError('original physics error')
    cleanup_resources((('session',fail('session failure')),
                       ('workspace',fail('workspace failure'))), primary=error)
    assert attempted == ['session failure','workspace failure']
    assert len(error.cleanup_errors) == 2
    assert 'workspace failure' in error.cleanup_errors[1]


def test_numerical_gauss_totals_preserve_measured_small_residual():
    from conductor_sim.femm5 import _SOURCE_CORE
    problem = ElectrostaticProblem((PointCharge(1e-9),), (
        SphericalShell(.1,.12,FixedNetCharge(1e-26),name='inner'),
        SphericalShell(.2,.22,FixedNetCharge(-1e-9),name='outer')))
    client = FakeClient()
    def measurements(name):
        return 0., {'inner':1e-26,'outer':-1e-9,_SOURCE_CORE:1e-9}[name]
    client.eo_getconductorproperties = measurements
    with Femm5Backend(session_factory=lambda: FakeSession(client)).solve(problem) as solution:
        assert solution.conductor_by_name('outer').outer_surface_charge_c == 1e-26
