"""Regression reproductions from the bounded post-M12 correctness review."""
from decimal import Decimal, localcontext
import json
from math import hypot, pi
from pathlib import Path
import subprocess
import sys

import pytest

from conductor_sim import (ConductingPlane, ElectrostaticProblem, FixedPotential,
                           GroundedPlaneBackend, ParallelPlateBackend, PointCharge,
                           Femm5PlaneBackend, Femm5ParallelPlateBackend)
from conductor_sim.geometry import (PlaneBoundary, SphereBoundary, CylinderSideBoundary,
                                    slice_geometry)
from conductor_sim.model import EPSILON_0
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.visualization import (SliceSpec, SamplingContext, sample_frame,
                                        sample_sweep, trace_field_line)
from test_femm5_adapter import FakeClient, FakeSession


def plates(charges=(), upper_anchor=(0., 0., .2), volts=20.):
    return ElectrostaticProblem(charges, (
        ConductingPlane(name='lower'),
        ConductingPlane(upper_anchor, (0., 0., -1.), FixedPotential(volts), 'upper'),
    ))


def plane_problem(position=(0., 0., .15)):
    return ElectrostaticProblem((PointCharge(1e-9, position),), (ConductingPlane(),))


def decimal_image(point):
    """Independent high-precision subtraction, not the implementation's algebra."""
    with localcontext() as ctx:
        ctx.prec = 70
        x, y, z = (Decimal(str(v)) for v in point)
        h = Decimal('.15')
        a = (x*x+y*y+(z-h)**2).sqrt()
        b = (x*x+y*y+(z+h)**2).sqrt()
        kq = Decimal('1e-9') / (4*Decimal(str(pi))*Decimal(str(EPSILON_0)))
        potential = kq*(1/a-1/b)
        vector = (kq*x*(1/a**3-1/b**3), kq*y*(1/a**3-1/b**3),
                  kq*((z-h)/a**3-(z+h)/b**3))
        return float(potential), tuple(float(v) for v in vector)


@pytest.mark.parametrize('point', [(1e8, 0., .07), (.05, 0., 1e-18),
                                    (.03, .04, .08), (1e6, 2e6, .4)])
def test_grounded_plane_potential_does_not_cancel_to_false_zero(point):
    solution = GroundedPlaneBackend().solve(plane_problem())
    exact, _ = decimal_image(point)
    assert exact > 0
    assert solution.potential_at(point) == pytest.approx(exact, rel=2e-13, abs=0.)


@pytest.mark.parametrize('point', [(1e8, 0., .07), (1e6, 2e6, .4), (.03, .04, .08)])
def test_grounded_plane_weak_tangential_field_is_preserved(point):
    solution = GroundedPlaneBackend().solve(plane_problem())
    _, exact = decimal_image(point)
    assert solution.electric_field_at(point) == pytest.approx(exact, rel=3e-13, abs=0.)


@pytest.mark.parametrize('boundary,a,b,expected', [
    (PlaneBoundary((0.,0.,0.), (0.,0.,1.)), (0.,0.,1e-12), (0.,0.,-1.), 1e-12/(1+1e-12)),
    (SphereBoundary((0.,0.,0.), .1), (.1-1e-12,0.,0.), (.2,0.,0.),
     (.1-(.1-1e-12))/(.2-(.1-1e-12))),
    (CylinderSideBoundary((0.,0.,0.), (0.,0.,1.), .1, 0., .2),
     (.1-1e-12,0.,.1), (.2,0.,.1), (.1-(.1-1e-12))/(.2-(.1-1e-12))),
])
def test_no_positive_near_start_boundary_crossing_is_discarded(boundary,a,b,expected):
    hit = boundary.first_hit(a,b)
    assert hit is not None
    assert hit == pytest.approx(expected, rel=3e-5, abs=0.)


@pytest.mark.parametrize('position', [(.03,.04,.12), (.4,-.2,.12)])
def test_default_planar_meridian_contains_the_charge_normal_line(position):
    p = plane_problem(position)
    spec = SliceSpec.meridian(p, resolution=9)
    x,y,out = spec.project(position)
    assert abs(out) < 1e-13
    assert spec.contains((x,y))
    assert spec.origin_m == pytest.approx((position[0],position[1],0.))


def test_parallel_plane_anchor_does_not_inflate_default_extent():
    ordinary = SliceSpec.meridian(plates(), resolution=9)
    equivalent = SliceSpec.meridian(plates(upper_anchor=(100.,-80.,.2)), resolution=9)
    assert ordinary.x_range_m == equivalent.x_range_m
    assert ordinary.y_range_m == equivalent.y_range_m


@pytest.mark.parametrize('charges', [
    (PointCharge(1e-9,(0.,0.,.19999)),),
    (PointCharge(1e-9,(0.,0.,.10)),PointCharge(-1e-9,(0.,0.,.100001))),
])
def test_display_masks_do_not_cross_electrodes_or_other_active_sites(charges):
    p = plates(charges)
    context = context_for_solution(p, ParallelPlateBackend().solve(p))
    for m in context.masks:
        assert m.radius_m < min(c.signed_distance_m(m.center_m) for c in p.conductors)
        for other in context.masks:
            if other is not m:
                assert m.radius_m+other.radius_m < hypot(*(a-b for a,b in zip(m.center_m,other.center_m)))


def test_cancelled_sites_do_not_change_active_display_exclusions():
    charge = PointCharge(1e-9,(0.,0.,.10))
    ordinary = plates((charge,))
    augmented = plates((charge,PointCharge(1e-9,(0.,0.,1e-9)),PointCharge(-1e-9,(0.,0.,1e-9))))
    backend = ParallelPlateBackend()
    a = context_for_solution(ordinary,backend.solve(ordinary))
    b = context_for_solution(augmented,backend.solve(augmented))
    assert a.masks == b.masks


def test_charge_free_capacitor_has_automatic_field_traces_but_zero_field_does_not():
    b = ParallelPlateBackend()
    frame = sample_sweep([('bias',plates())],b,resolution=13,line_steps=120)['frames'][0]
    assert frame['field_lines']
    assert frame['line_termination_counts'].get('surface',0) > 0
    for line in frame['field_lines']:
        assert line['points_m'][-1][1] < line['points_m'][0][1]  # E toward lower potential
    zero = sample_sweep([('zero',plates(volts=0.))],b,resolution=13,line_steps=20)['frames'][0]
    assert zero['field_lines'] == []


@pytest.mark.parametrize('backend,p', [
    (Femm5ParallelPlateBackend, plates()),
    (Femm5PlaneBackend, plane_problem()),
])
def test_wide_femm_density_profiles_are_masked_not_sampled_outside_domain(backend,p):
    session = FakeSession(FakeClient())
    solution = backend(session_factory=lambda:session).solve(p)
    try:
        context = context_for_solution(p,solution)
        frame = sample_frame(context, SliceSpec(x_range_m=(-2.,2.),y_range_m=(-.1,.3),resolution=9),
                             seeds=[],profile_samples=21)
        for profile in frame['profiles']:
            values = profile['sigma_c_per_m2']
            assert any(v is None for v in values)
            assert any(v is not None for v in values)
        json.dumps(frame,allow_nan=False)
    finally:
        solution.close()
    assert session.closes == 1


def test_femm_density_offset_outside_hemisphere_is_an_explicit_domain_exclusion():
    p = plane_problem()
    sol = Femm5PlaneBackend(session_factory=lambda:FakeSession(FakeClient())).solve(p)
    try:
        # Surface point is in the disk, but the one-sided density probe is not.
        r = sol.outer_radius_m*(1-1e-7)
        spec = SliceSpec(x_range_m=(-r/2**.5,r/2**.5),
                         y_range_m=(-r/2**.5,r/2**.5),resolution=9)
        frame = sample_frame(context_for_solution(p,sol),spec,seeds=[],profile_samples=9)
        assert frame['profiles'][0]['sigma_c_per_m2'][0] is None
    finally:
        sol.close()


def test_density_failure_is_not_silently_replaced_with_missing_data():
    p = plates()
    context = context_for_solution(p,ParallelPlateBackend().solve(p))
    def broken(*args):
        raise RuntimeError('density solver failed')
    context.density_at = broken
    with pytest.raises(RuntimeError,match='density solver failed'):
        sample_frame(context,SliceSpec.meridian(p,resolution=9),seeds=[])


@pytest.mark.parametrize('height,metal', [(-.1,True), (.1,False), (.3,True)])
def test_slice_parallel_to_a_plate_describes_its_filled_metal_region(height,metal):
    spec = SliceSpec(origin_m=(0.,0.,height),u=(1.,0.,0.),v=(0.,1.,0.),resolution=9)
    geometries = [slice_geometry(c,spec) for c in plates().conductors]
    assert all(not g['visible'] for g in geometries)
    assert any(g.get('fills_metal',False) for g in geometries) == metal


@pytest.mark.parametrize('cleanup_exception',[RuntimeError, KeyboardInterrupt])
def test_sweep_preserves_primary_exception_and_secondary_cleanup_diagnostics(cleanup_exception):
    failure = ValueError('primary sampling error')
    class Owned:
        conductor_results = ()
        def potential_at(self,point): raise failure
        def close(self): raise cleanup_exception('secondary cleanup error')
    class Backend:
        def solve(self,problem): return Owned()
    caught = None
    try:
        sample_sweep([('bad',plates())],Backend(),plane=SliceSpec(resolution=9),seeds=[])
    except BaseException as error:
        caught = error
    assert caught is failure
    assert 'secondary cleanup error' in repr(getattr(failure,'cleanup_errors',()))


def test_parallel_plate_example_writes_the_advertised_json_sidecar(tmp_path):
    output = tmp_path/'plates.html'
    script = Path(__file__).resolve().parents[1]/'examples'/'parallel_plates.py'
    subprocess.run([sys.executable,str(script),'--resolution','9','--image-order','1',
                    '--output',str(output)],check=True,capture_output=True,text=True)
    assert output.exists()
    raw = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
    assert raw['schema'] == 'conductor_sim.visualization.v1'
    assert len(raw['frames']) == 3
