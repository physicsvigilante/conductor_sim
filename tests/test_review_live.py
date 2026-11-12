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
