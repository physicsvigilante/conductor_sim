"""JEE12 p27 developer-only checks must distinguish ideal source limit and true finite model."""
from copy import deepcopy
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from tools.jee12_p27_source_probe import check
from tools.m36c_jee_conductor_replay import _p27_original_A_B
from tools.m36c_jee_conductor_reference import original_p27_AB


def test_jee12_p27_original_symbolic_source_and_finite_metal_are_distinct():
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as temp:
        data = check(root, Path(temp))
        card_file = Path(temp) / data["existing_source_backed_replay"]["source_replay_visible_html_filename"]
        assert card_file.is_file()
        assert 'id="p27-original-A-B"' in card_file.read_text(encoding="utf-8")
    assert data["source"]["originally_asked"] == ["V(A)", "V(B)"]
    assert data["original_textbook_was_not_numeric"] is True
    assert data["browser_against_this_evaluation_newly_independently_checked"] is False
    assert data["original_problem_answer_independently_reviewed_for_jee12"] is False
    assert len(data["teacher_finite_thickness_signed_cases"]) == 3
    for case in data["teacher_finite_thickness_signed_cases"]:
        for label in ("va", "vb"):
            native = case[f"native_finite_metal_{label}_v"]
            ref = case[f"independent_finite_metal_{label}_v"]
            assert isclose(native, ref, rel_tol=4e-13, abs_tol=2e-13)
    # A nonzero actual metal thickness must not silently become the printed ideal value.
    for label in ("va", "vb"):
        case=data["teacher_finite_thickness_signed_cases"][0]
        assert abs(case[f"finite_minus_ideal_{label}_v"]) > 1e-6


def test_jee12_p27_existing_independent_oracle_rejects_displaced_source_for_original():
    with TemporaryDirectory() as temp:
        card = _p27_original_A_B(Path(temp))
        assert original_p27_AB(card)["checked_source_observations"] == 5
        forged = deepcopy(card)
        forged["authored_states"][0]["charges"][0]["position_m"] = [0., 0., .02]
        with pytest.raises(AssertionError, match="central q must be FIRST"):
            original_p27_AB(forged)
        forged = deepcopy(card)
        forged["original_formula_limit"]["V_B_zero_thickness_v"] = False
        with pytest.raises(AssertionError, match="nonnumeric/nonfinite|saved="):
            original_p27_AB(forged)
