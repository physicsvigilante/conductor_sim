"""Potential interpolation and progressive trace-pool regressions; no FEMM needed."""
from copy import deepcopy
from math import hypot
import json

import pytest
from conductor_sim import (ConductingPlane, ElectrostaticProblem, FixedPotential,
                           ParallelPlateBackend, PointCharge, Grounded,
                           SphericalShell, NestedShellBackend)
from conductor_sim.potential_display import potential_interpolation, potential_colour_scale
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.visual_seeds import seed_pool, distinct_pool, _progressive
from conductor_sim.visualization import SliceSpec, SamplingContext, sample_frame, sample_sweep
from conductor_sim.viewer import render_html


def slab(charges=(), voltage=20.):
    return ElectrostaticProblem(tuple(charges), (
        ConductingPlane(name='lower'),
        ConductingPlane((0.,0.,.2),(0.,0.,-1.),FixedPotential(voltage),'upper')))


def slab_frame(charges=(), voltage=20., seeds=()):
    return sample_sweep([('slab',slab(charges,voltage))],ParallelPlateBackend(image_order=8),
                        plane=SliceSpec(x_range_m=(-.3,.3),y_range_m=(-.1,.3),resolution=13),
                        seeds=seeds,line_steps=130)['frames'][0]


def test_safe_vacuum_cells_and_plane_edge_voltages_are_not_clamped_or_blurred():
    frame = slab_frame()
    stencil = frame['potential_interpolation']
    grid = frame['grid']
    assert any(any(row) for row in stencil['cells'])
    assert stencil['boundary']  # top/bottom cross the sampled rows
    for item in stencil['boundary']:
        for triangle in item['triangles']:
            for x,y,value in triangle:
                assert -1e-14 <= y <= .2+1e-14
                assert value == pytest.approx(100*y, abs=1e-10)
    for j,row in enumerate(stencil['cells']):
        for i,valid in enumerate(row):
            if valid:
                assert grid['y_m'][j] >= -1e-14 and grid['y_m'][j+1] <= .2+1e-14


def _one_cell():
    return {'plane':{'origin_m':[0,0,0],'u':[1,0,0],'v':[0,0,1]},
            'geometry':[], 'source_masks':[], 'conductor_results':[],
            'display_clip_v':2.,
            'grid':{'x_m':[0,1],'y_m':[0,1],'potential_v':[[0.,1.],[1.,2.]],
                    'region':[['vacuum','vacuum'],['vacuum','vacuum']]}}


@pytest.mark.parametrize('kind',['source-mask','outside-domain'])
def test_missing_node_is_never_replaced_by_neighbour_or_zero(kind):
    frame=_one_cell()
    frame['grid']['potential_v'][0][0]=None
    frame['grid']['region'][0][0]=kind
    display=potential_interpolation(frame)
    assert display['cells']==[[False]]
    assert display['boundary']==[]  # both original triangles share this null node


def test_entirely_missing_and_wholly_metal_cells_stay_blank():
    frame=_one_cell()
    frame['grid']['potential_v']=[[None,None],[None,None]]
    assert potential_interpolation(frame)['boundary']==[]
    frame=_one_cell()
    frame['geometry']=[{'kind':'plane','visible':False,'fills_metal':True}]
    assert potential_interpolation(frame)['cells']==[[False]]
    assert potential_interpolation(frame)['boundary']==[]


def test_subcell_source_mask_without_null_nodes_is_not_bridged():
    frame=_one_cell()
    frame['source_masks']=[{'center_m':[.5,0,.5],'radius_m':.02}]
    display=potential_interpolation(frame)
    assert display['cells']==[[False]]
    assert display['boundary']==[]  # mask intersects both triangles on diagonal


def test_thin_spherical_shell_between_vacuum_nodes_is_not_bridged():
    frame=_one_cell()
    frame['geometry']=[{'kind':'sphere','visible':True,'center_m':[.5,.5],
                        'inner_radius_m':.2,'outer_radius_m':.20001}]
    display=potential_interpolation(frame)
    assert display['cells']==[[False]]
    assert display['boundary']==[]


def test_full_vacuum_cell_uses_convex_bilinear_values():
    frame=_one_cell()
    display=potential_interpolation(frame)
    assert display=={'method':'masked-bilinear-with-plane-boundaries','cells':[[True]],'boundary':[]}


def test_colour_range_covers_stored_values_not_old_percentile_saturation():
    a,b=_one_cell(),_one_cell()
    b['grid']['potential_v'][0][1]=-400.
    a['grid']['potential_v'][0][0]=1000.
    scale=potential_colour_scale([a,b])
    assert scale['max_abs_v']==1000.
    assert 0<scale['softening_v']<=1000.
    a['conductor_results']=[{'name':'fixed','potential_v':1500.}]
    assert potential_colour_scale([a,b])['max_abs_v']==1500.
    a['grid']['potential_v']=[[0.,0.],[0.,0.]];a['conductor_results']=[]
    zero=potential_colour_scale([a])
    assert zero['max_abs_v']>0 and zero['softening_v']>0


def test_progressive_sequence_fills_both_halves_then_quarters_without_repeats():
    values=list(_progressive())
    assert values[:4]==[.5,.25,.75,.125]
    assert len(values)==32==len(set(values))
    assert all(0<x<1 for x in values)


def test_prefix_balances_both_signed_sources_and_both_plates():
    p=slab((PointCharge(1e-9,(0,0,.065)),PointCharge(-.5e-9,(0,0,.14))),0.)
    backend=ParallelPlateBackend(image_order=8)
    plane=SliceSpec.meridian(p,extent_m=.29,resolution=21)
    context=context_for_solution(p,backend.solve(p))
    first=seed_pool(context,plane)
    assert first==seed_pool(context,plane)
    assert 10<len(first)<=96
    assert {s[2] for s in first[:4]}=={'charge:0','charge:1','face:lower:outer','face:upper:outer'}
    assert all(context.region(plane.world(s[0]))=='vacuum' for s in first)
    frame=sample_frame(context,plane,line_steps=180)
    assert frame['field_line_pool']['kind']=='progressive-families'
    assert frame['field_line_pool']['available']==len(frame['field_lines'])
    assert len(frame['field_lines'])>40
    assert {l['seed_family'] for l in frame['field_lines'][:8]}=={s[2] for s in first[:4]}
    for line in frame['field_lines']:
        for a,b in zip(line['points_m'],line['points_m'][1:]):
            assert context.first_hit(plane.world(a),plane.world(b)) is None


def test_charge_free_capacitor_has_spread_lines_without_two_duplicate_faces():
    frame=slab_frame(seeds=None)
    lines=frame['field_lines']
    assert 20<=len(lines)<=32
    assert {line['seed_family'] for line in lines}=={'face:lower:outer'}
    assert min(line['points_m'][0][0] for line in lines[:4])<0<max(line['points_m'][0][0] for line in lines[:4])
    assert all(line['points_m'][0][1]>line['points_m'][-1][1] for line in lines)
    assert slab_frame(voltage=0,seeds=None)['field_lines']==[]


def test_coincident_canceled_sources_produce_no_phantom_seed_family():
    frame=slab_frame((PointCharge(1e-9,(0,0,.1)),PointCharge(-1e-9,(0,0,.1))),seeds=None)
    assert not frame['charges']
    assert all(not line['seed_family'].startswith('charge:') for line in frame['field_lines'])


def test_spherical_seed_pool_includes_charge_inner_and_outer_faces():
    p=ElectrostaticProblem((PointCharge(1e-9,(0,0,.03)),),
         (SphericalShell(.1,.12,Grounded(),name='shell'),))
    context=context_for_solution(p,NestedShellBackend().solve(p))
    seeds=seed_pool(context,SliceSpec.meridian(p,resolution=13))
    assert any(s[2]=='charge:0' for s in seeds)
    assert any(s[2]=='face:shell:inner' for s in seeds)
    # Grounded, shielded exterior has E=0: no fictitious exterior traces.
    assert not any(s[2]=='face:shell:outer' for s in seeds)


def test_explicit_seeds_and_empty_list_keep_their_existing_semantics():
    assert slab_frame(seeds=[])['field_lines']==[]
    frame=slab_frame(seeds=[((0,.1),1)])
    assert len(frame['field_lines'])==1
    assert frame['field_line_pool']['kind']=='authored'
    assert 'seed_family' not in frame['field_lines'][0]


def test_near_duplicate_paths_removed_without_moving_retained_paths():
    a={'points_m':[(0.,0.),(.1,.1),(.2,.2)]}
    b={'points_m':[(.05,0.),(.15,.1),(.25,.2)]}
    assert distinct_pool([a,deepcopy(a),b],SliceSpec())==[a,b]


def test_old_bundle_render_is_compatible_and_does_not_mutate_original():
    frame=slab_frame()
    frame.pop('potential_interpolation');frame.pop('field_line_pool')
    bundle={'schema':'conductor_sim.visualization.v1','color_limit_v':20.,'frames':[frame]}
    original=deepcopy(bundle)
    html=render_html(bundle)
    assert bundle==original
    assert 'id="line-density"' in html
    assert 'potentialSample' in html
    assert 'asinh colour scale' in html


def test_display_metadata_does_not_change_solver_grid_values():
    frame=slab_frame()
    before=deepcopy(frame['grid'])
    potential_interpolation(frame);potential_colour_scale([frame])
    assert frame['grid']==before
    json.dumps(frame,allow_nan=False)
