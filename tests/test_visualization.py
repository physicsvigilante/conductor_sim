"""Sampling, geometry, serialization and ownership tests; no FEMM required."""
from math import hypot, isfinite, pi
from pathlib import Path
import json
import pytest
from conductor_sim import ElectrostaticProblem, PointCharge, SphericalShell, Grounded, FixedNetCharge, NestedShellBackend
from conductor_sim.solver import ConductorResult
from conductor_sim.visualization import (SliceSpec,SamplingContext,SourceMask,trace_field_line,sample_frame,sample_sweep,write_visualization,contour_segments)
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.viewer import render_html


class Uniform:
    conductor_results=()
    def potential_at(self,p): return -p[0]
    def electric_field_at(self,p): return (1.,0.,0.)


def problem(q=1e-9):
    return ElectrostaticProblem((PointCharge(q,(0,0,.03)),),
        (SphericalShell(.1,.12,FixedNetCharge(0),name='inner'),SphericalShell(.2,.22,Grounded(),name='outer')))


def plane(n=13):
    return SliceSpec(x_range_m=(-.3,.3),y_range_m=(-.3,.3),resolution=n)


def test_plane_round_trip_and_rotation():
    spec=SliceSpec((2,3,4),(0,1,0),(1,0,0),resolution=9)
    assert spec.project(spec.world((.2,-.1)))==pytest.approx((.2,-.1,0))
    for u,v in [((0,0,0),(0,0,1)),((1,0,0),(1,0,0))]:
        with pytest.raises(ValueError):SliceSpec(u=u,v=v)


@pytest.mark.parametrize('settings',[{'resolution':8},{'resolution':202},{'resolution':12.5},
 {'x_range_m':(1,1)},{'y_range_m':(float('nan'),3)}])
def test_bad_slice_parameters(settings):
    with pytest.raises(ValueError):SliceSpec(**settings)


def test_trace_is_ordered_with_E_in_both_integration_directions():
    c=SamplingContext(ElectrostaticProblem(),Uniform(),'uniform')
    for direction in (1,-1):
        line=trace_field_line(c,plane(),(0,0),direction=direction,step_m=.02,max_steps=20)
        xs=[p[0] for p in line['points_m']]
        assert all(b>a for a,b in zip(xs,xs[1:]))
        assert line['termination']=='slice-boundary'


def test_large_step_cannot_tunnel_through_thin_metal():
    p=ElectrostaticProblem(conductors=(SphericalShell(.1,.10001,Grounded()),))
    c=SamplingContext(p,Uniform(),'uniform')
    line=trace_field_line(c,plane(),(.05,0),step_m=.16,max_steps=4)
    assert line['termination']=='surface'
    assert .09999<line['points_m'][-1][0]<.1
    assert all(hypot(*v)<.1 for v in line['points_m'])


def test_swept_segment_detects_source_core_between_endpoints():
    c=SamplingContext(ElectrostaticProblem(),Uniform(),'uniform',
                      masks=(SourceMask((.1,0,0),.01,'numerical-core'),))
    line=trace_field_line(c,plane(),(0,0),step_m=.2,max_steps=4)
    assert line['termination']=='source-mask'
    assert line['points_m'][-1][0]==pytest.approx(.09,rel=1e-6)


def test_domain_mask_and_non_planar_field_are_explicit():
    c=SamplingContext(ElectrostaticProblem(),Uniform(),'uniform',domain_radius_m=.15)
    assert trace_field_line(c,plane(),(0,0),step_m=.2)['termination']=='outside-domain'
    class Tilt(Uniform):
        def electric_field_at(self,p):return (1,1,0)
    c.solution=Tilt()
    line=trace_field_line(c,plane(),(0,0))
    assert line['termination']=='out-of-plane'
    assert len(line['points_m'])==1


def test_zero_field_stops_and_step_count_is_bounded():
    class Zero(Uniform):
        def electric_field_at(self,p):return (0,0,0)
    c=SamplingContext(ElectrostaticProblem(),Zero(),'zero')
    assert trace_field_line(c,plane(),(0,0))['termination']=='zero-field'
    c.solution=Uniform()
    line=trace_field_line(c,plane(),(0,0),step_m=.001,max_steps=3)
    assert line['termination']=='step-limit' and len(line['points_m'])==4


def test_grid_matches_underlying_solution_and_profile_methods():
    p=problem();sol=NestedShellBackend().solve(p)
    c=context_for_solution(p,sol)
    f=sample_frame(c,plane(),seeds=[],profile_samples=9)
    assert len(f['profiles'])==4
    assert f['backend']=='NestedShellSolution'
    assert f['density_method']=='exact analytical surface density'
    for j,y in enumerate(f['grid']['y_m']):
        for i,x in enumerate(f['grid']['x_m']):
            value=f['grid']['potential_v'][j][i]
            if value is not None:
                assert value==pytest.approx(sol.potential_at((x,0,y)))
    json.dumps(f,allow_nan=False)


def test_source_masks_are_null_not_zero_or_interpolated():
    p=ElectrostaticProblem((PointCharge(1e-9),),(SphericalShell(.1,.12,Grounded()),))
    c=context_for_solution(p,NestedShellBackend().solve(p))
    f=sample_frame(c,plane(),seeds=[],profile_samples=9)
    assert f['grid']['potential_v'][6][6] is None
    assert f['grid']['electric_field_v_per_m'][6][6] is None
    assert f['grid']['region'][6][6]=='source-mask'
    for contour in f['contours']:
        for segment in contour['segments_m']:
            assert c.first_hit(*(plane().world(x) for x in segment)) is None


def test_linear_potential_contours_and_no_contours_through_mask():
    c=SamplingContext(ElectrostaticProblem(),Uniform(),'uniform')
    f=sample_frame(c,plane(),seeds=[],profile_samples=9)
    contours=contour_segments(f['grid'],[.05],c,plane())
    assert contours[0]['segments_m']
    for segment in contours[0]['segments_m']:
        for x,y in segment:assert x==pytest.approx(-.05)


def test_sweep_closes_numerical_owners_on_success_and_sampling_failure():
    objects=[]
    class Owned(Uniform):
        def __init__(self,fail):self.closed=False;self.fail=fail
        def potential_at(self,p):
            if self.fail:raise RuntimeError('real solver failed')
            return super().potential_at(p)
        def close(self):self.closed=True
    class Backend:
        def __init__(self,fail=False):self.fail=fail
        def solve(self,p):
            s=Owned(self.fail);objects.append(s);return s
    p=problem()
    sample_sweep([('one',p),('two',p)],Backend(),plane=plane(),seeds=[])
    assert all(s.closed for s in objects)
    with pytest.raises(RuntimeError,match='real solver failed'):
        sample_sweep([('fail',p)],Backend(True),plane=plane(),seeds=[])
    assert objects[-1].closed


def test_nonfinite_solver_result_is_failure_not_a_hole():
    class Bad(Uniform):
        def potential_at(self,p):return float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        sample_frame(SamplingContext(ElectrostaticProblem(),Bad(),'bad'),plane(),seeds=[])


def test_html_escapes_embedded_data_and_does_not_require_network(tmp_path):
    p=problem();label='</script><script>window.PWNED=1</script>'
    bundle=sample_sweep([(label,p)],NestedShellBackend(),plane=plane(),seeds=[])
    html=render_html(bundle,title='<bad>')
    assert label not in html and '\\u003c/script\\u003e' in html
    assert '<title>&lt;bad&gt;</title>' in html
    assert '<script src=' not in html and '<link href=' not in html
    path=write_visualization(bundle,tmp_path/'view.html')
    assert path.is_file() and not list(tmp_path.glob('*.tmp'))


def test_sweep_uses_shared_color_scale_and_actual_distinct_states():
    b=sample_sweep([('q',problem()),('2q',problem(2e-9))],NestedShellBackend(),plane=plane(),seeds=[])
    a,c=b['frames'];assert b['color_limit_v']==max(a['display_clip_v'],c['display_clip_v'])
    assert c['grid']['potential_v'][6][7]==pytest.approx(2*a['grid']['potential_v'][6][7])
    assert a['plane']==c['plane']


def test_failed_export_leaves_existing_file_unchanged(tmp_path):
    path=tmp_path/'view.html';path.write_text('old')
    with pytest.raises(ValueError):write_visualization({'schema':'bad'},path)
    assert path.read_text()=='old'


def test_solver_surface_tolerance_terminates_trace_not_export():
    p=ElectrostaticProblem((PointCharge(1e-9),),(SphericalShell(.1,.12,Grounded()),))
    c=context_for_solution(p,NestedShellBackend().solve(p))
    line=trace_field_line(c,plane(),(.1*(1-5e-13),0))
    assert line['termination']=='surface'


def test_bound_infinite_authored_iterables_before_sampling():
    from itertools import repeat
    with pytest.raises(ValueError,match='24 cases'):
        sample_sweep(repeat(('infinite',problem())),NestedShellBackend())
    c=SamplingContext(ElectrostaticProblem(),Uniform(),'uniform')
    with pytest.raises(ValueError,match='96 seed'):
        sample_frame(c,plane(),seeds=repeat(((0,0),1)))


def test_primary_failure_preserved_if_close_also_fails():
    class Owned(Uniform):
        def potential_at(self,p):raise RuntimeError('primary failure')
        def close(self):raise RuntimeError('secondary cleanup')
    class Backend:
        def solve(self,p):return Owned()
    with pytest.raises(RuntimeError,match='primary failure'):
        sample_sweep([('broken',problem())],Backend(),plane=plane(),seeds=[])


def test_template_tokens_in_user_content_are_not_replaced_recursively():
    bundle=sample_sweep([('__TITLE__',problem())],NestedShellBackend(),plane=plane(),seeds=[])
    html=render_html(bundle,title='__DATA__')
    assert '<title>__DATA__</title>' in html
    assert '"label":"__TITLE__"' in html
