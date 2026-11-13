"""Post-M9 export regression with actual FEMM, not a fake field client."""
import os
import pytest
from conductor_sim import ElectrostaticProblem, PointCharge, SolidSphere, Grounded, Femm5ExternalSphereBackend
from conductor_sim.femm5 import Femm5Session
from conductor_sim.visualization import sample_sweep

pytestmark = [pytest.mark.femm5, pytest.mark.skipif(not os.environ.get('FEMM5_BINDING'),
                                                   reason='real FEMM 5 binding not configured')]


def test_real_femm_auto_view_seed_replay_and_owned_session_cleanup():
    sessions = []
    class TrackedSession(Femm5Session):
        def __init__(self):
            super().__init__()
            self.closes = 0
        def close(self):
            self.closes += 1
            super().close()
    def factory():
        session = TrackedSession()
        sessions.append(session)
        return session
    p = ElectrostaticProblem((PointCharge(1e-9, (0,0,.27)),),
                             (SolidSphere(.12, Grounded()),))
    bundle = sample_sweep([('first', p), ('second', p)],
                         Femm5ExternalSphereBackend(session_factory=factory),
                         resolution=13, line_steps=30, seeds=iter([((.16,0),1)]))
    for frame in bundle['frames']:
        assert len(frame['field_lines']) == 1
        assert sum(frame['line_termination_counts'].values()) == 1
        assert frame['plane']['y_range_m'][1] > .27
        measured = frame['conductor_results'][0]
        assert measured['potential_v'] == pytest.approx(0, abs=.1)
        assert measured['net_charge_c'] == pytest.approx(-1e-9*.12/.27, rel=.14, abs=1e-11)
    assert len(sessions) == 2
    assert all(s.closes == 1 and s._client is None for s in sessions)


@pytest.mark.parametrize('pair', [False, True])
def test_real_femm_wide_planar_export_masks_density_outside_mesh(pair):
    from conductor_sim import (ConductingPlane, FixedPotential, Femm5PlaneBackend,
                               Femm5ParallelPlateBackend)
    from conductor_sim.visualization import SliceSpec
    if pair:
        p = ElectrostaticProblem(conductors=(ConductingPlane(name='lower'),
            ConductingPlane((0.,0.,.2),(0.,0.,-1.),FixedPotential(20.),'upper')))
        backend = Femm5ParallelPlateBackend()
    else:
        p = ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.15)),),
                                 (ConductingPlane(),))
        backend = Femm5PlaneBackend()
    frame = sample_sweep([('wide actual FEMM',p)],backend,
                        plane=SliceSpec(x_range_m=(-2.,2.),y_range_m=(-.1,.3),resolution=9),
                        seeds=[],line_steps=20)['frames'][0]
    assert any(r=='outside-domain' for row in frame['grid']['region'] for r in row)
    for profile in frame['profiles']:
        assert any(v is None for v in profile['sigma_c_per_m2'])
        assert any(v is not None for v in profile['sigma_c_per_m2'])
    if pair:
        assert frame['profiles'][0]['sigma_c_per_m2'][45] == pytest.approx(
            -100*p.permittivity_f_per_m,rel=.28,abs=2.5e-10)


def test_real_femm_translated_plane_default_view_contains_the_source():
    from conductor_sim import ConductingPlane, Femm5PlaneBackend
    p = ElectrostaticProblem((PointCharge(1e-9,(.03,.04,.15)),),
                             (ConductingPlane(),))
    frame = sample_sweep([('translated source',p)],Femm5PlaneBackend(),
                        resolution=13,line_steps=80)['frames'][0]
    assert frame['plane']['origin_m'] == pytest.approx((.03,.04,0.))
    assert abs(frame['charges'][0]['slice_position_m'][2]) < 1e-12
    assert frame['field_lines']
