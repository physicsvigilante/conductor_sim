"""Submission contract for independent evaluation of an authored JEE scenario."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from conductor_sim import PositionSweepSpec


@dataclass(frozen=True, slots=True)
class Submission:
    capability: Literal['analytical-only', 'unsupported']
    backend: Literal['plane', 'sphere', 'plates'] | None = None
    sweep: PositionSweepSpec | None = None
    reason: str = ''
