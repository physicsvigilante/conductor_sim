"""JEE12 source p69: authentic spherical Gauss totals and signed local pattern."""
from math import isclose
from pathlib import Path
import pytest

from tools.jee12_p69_source_probe import _model, check, DIRECTIONS
from conductor_sim.spherical_multicavity import solve_spherical_sigma, NamedSourcePosition, SphericalVoid
from conductor_sim.solver import UnsupportedProblem


def test_jee12_p69_original_figure_inner_and_outer_charge_distributions():
    data=check(Path(__file__).resolve().parents[1])
    assert data["source"]["source_given_metal_charge"] == "3q"
    assert data["source"]["original_qualitative_results"] == [
        "inner -q uniform if centered", "inner -q nonuniform if displaced",
        "outer +4q uniform in both"]
    assert data["original_source_problem_newly_independently_reviewed"] is False
    assert data["newly_independently_browser_verified"] is False
    assert data["no_physical_elapsed_time_or_current_claimed"] is True
    assert len(data["teacher_signed_static_cases"]) == 2
    for case in data["teacher_signed_static_cases"]:
        q=case["teacher_q_c"]
        assert isclose(case["native_inner_face_charge_c"], -q, rel_tol=1e-13)
        assert isclose(case["native_outer_face_charge_c"], 4*q, rel_tol=1e-13)
        assert case["inner_centered_uniform"] is True
        assert case["inner_shifted_nonuniform"] is True
        assert len(case["discrete_signed_samples"]) == len(DIRECTIONS)


def test_jee12_p69_source_position_cannot_move_outside_qualified_cavity():
    with pytest.raises(UnsupportedProblem, match="outside/on/too close"):
        solve_spherical_sigma(_model(1e-9,3e-9,(.9,0.,0.)))
    model=_model(1e-9,3e-9,(0.,0.,0.))
    assert model.geometry.voids[0].positions[0].position_m==(0.,0.,0.)
    before=solve_spherical_sigma(model)
    after=solve_spherical_sigma(_model(1e-9,3e-9,(.2,.1,.1)))
    assert isclose(before.outer_surface_charge_density(DIRECTIONS[0]),
                   after.outer_surface_charge_density(DIRECTIONS[1]),rel_tol=1e-13)
