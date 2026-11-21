"""JEE12 p26 source-anchored numerical case is developer evidence only."""
from math import isclose
from pathlib import Path

from tools.jee12_p26_source_probe import check


def test_jee12_p26_solid_source_native_vs_independent_equations():
    data = check(Path(__file__).resolve().parents[1])
    assert data["original_source"]["original_answer_symbolic"] == ["kQ/R", "2kQ/(3R)"]
    assert data["original_problem_answer_verified_by_independent_reviewer"] is False
    assert data["browser_case_newly_checked"] is False
    assert data["physical_elapsed_time"] is False
    assert len(data["teacher_numeric_cases"]) == 3
    for case in data["teacher_numeric_cases"]:
        assert isclose(case["native_v_half_r"], case["independently_expected_v_half_r"], rel_tol=3e-14)
        assert isclose(case["native_v_three_half_r"], case["independently_expected_v_three_half_r"], rel_tol=3e-14)
