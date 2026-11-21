from fractions import Fraction

import pytest

from conductor_sim.affine_fields import AffineFieldAssembly, ConservativeAffineField
from conductor_sim.path_barriers import StraightPathEnergyProfile, straight_path_energy_profile


def test_direct_profile_preserves_rational_peak_and_rejects_invalid_slopes():
    assert StraightPathEnergyProfile(1, -2).barrier().peak_fraction == Fraction(1, 3)
    for invalid in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            StraightPathEnergyProfile(Fraction(1, 2), invalid)


def test_guide_profile_integrates_coupled_field_and_rejects_extrapolation():
    source = ConservativeAffineField(((0, 3, 0), (3, 0, 0), (0, 0, 0)), reference_potential_v=1e100)
    profile = straight_path_energy_profile(AffineFieldAssembly((source,)), -2, (1, 2, 3), (4, -1, 5))
    for s in (0, Fraction(1, 3), Fraction(1, 2), 1):
        # q[V(x,y)-V(1,2)] = 6(xy-2).
        assert profile.energy_change_j(s) == pytest.approx(6*((1+3*s)*(2-3*s)-2))
    assert profile.barrier().peak_fraction == Fraction(1, 6)
    for outside in (-0.1, 1.01):
        with pytest.raises(ValueError, match="fraction"):
            profile.energy_change_j(outside)
