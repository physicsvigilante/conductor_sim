"""Small authoring-only submission interface; never changes the physics engine."""
from dataclasses import dataclass
from typing import Literal

from conductor_sim import ElectrostaticProblem
from conductor_sim.model import Vector3


@dataclass(frozen=True, slots=True)
class Probe:
    state: int
    label: str
    kind: Literal['potential','field_x','field_y','field_z','shell_voltage',
                  'shell_net_charge','inner_total','outer_total']
    point_m: Vector3 | None = None
    conductor: str | None = None

    @property
    def signature(self):
        return (self.state,self.kind,self.point_m,self.conductor)


@dataclass(frozen=True, slots=True)
class Submission:
    """User-written typed physics and requested measurements, or a refusal."""
    capability: Literal['both','analytical-only','unsupported']
    states: tuple[tuple[str, ElectrostaticProblem], ...] = ()
    probes: tuple[Probe,...] = ()
    reason: str = ''
