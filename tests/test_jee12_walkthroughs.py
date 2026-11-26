"""Source-faithful native static JEE walkthroughs with independent SI refusals."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tools.jee12_walkthroughs import generate, energy_record, work_record
from tools.jee12_walkthrough_reference import verify, verify_work, verify_energy


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("jee12-two-original-walkthroughs")
    root = Path(__file__).resolve().parents[1]
    manifest = generate(root, out)
    return root, out, manifest


def test_native_original_two_case_source_physics_and_standalone_reference(generated):
    root, out, manifest = generated
    assert [x["case_id"] for x in manifest["pages"]] == [
        "p01-single-08", "p03-single-27"]
    result = verify(out, root)
    assert result["independent_si"] == "PASS"
    assert result["frozen_jee12_original_answer_count_changed"] is False
    import json
    from math import isclose
    work = json.loads((out / "p1-external-work.json").read_text(encoding="utf-8"))
    energy = json.loads((out / "p3-solid-shell-energy.json").read_text(encoding="utf-8"))
    assert isclose(work["native"]["external_work_j"]/1.602176634e-19, 40., rel_tol=1e-12)
    assert isclose(work["native"]["field_work_j"]/1.602176634e-19, -40., rel_tol=1e-12)
    assert isclose(work["teacher_hypothetical"]["native_external_work_j"]/1.602176634e-19,
                   80., rel_tol=1e-12)
    assert 8.98 < energy["native"]["energy_released_j"] < 9.
    assert energy["source_given"]["solid_radius_m"] == 0.1
    assert energy["teacher_model_qualification"]["actual_inner_body"].startswith("SOLID")
    assert isclose(energy["teacher_hypothetical"]["native_energy_released_j"],
                   energy["native"]["energy_released_j"]/4, rel_tol=2e-11)


@pytest.mark.parametrize("field,value", [
    ("field_work_j", 40.*1.602176634e-19),
    ("external_work_j", -40.*1.602176634e-19),
    ("end_potential_v", 40.),
])
def test_bad_original_work_observations_refused(field, value):
    original = work_record()
    forged = deepcopy(original)
    forged["native"][field] = value
    with pytest.raises(AssertionError):
        verify_work(forged)


@pytest.mark.parametrize("group,key,value", [
    ("source_given", "uniform_field_x_v_m", 20.),
    ("teacher_hypothetical", "changed_field_x_v_m", 10.),
    ("teacher_hypothetical", "role", "SOURCE ORIGINAL"),
    ("derived", "horizontal_displacement_m", 3.),
    ("printed_answer", "option", 3),
])
def test_original_teacher_work_attribution_and_geometry_refused(group, key, value):
    forged = deepcopy(work_record())
    forged[group][key] = value
    with pytest.raises(AssertionError):
        verify_work(forged)


@pytest.mark.parametrize("path,value", [
    (("source_given", "solid_radius_m"), 0.05),
    (("source_given", "initial_solid_charge_c"), 10e-6),
    (("teacher_model_qualification", "actual_inner_body"), "HOLLOW shell proxy"),
    (("native", "before", "shell_inner_face_c"), 20e-6),
    (("native", "after", "solid_net_charge_c"), 20e-6),
    (("native", "energy_released_j"), 0.),
    (("native", "after", "gap_energy_j"), 1.),
    (("native", "before", "exterior_energy_j"), 0.),
    (("source_given", "initial_shell_net_charge_c"), 0.),
    (("model_assumptions", "concentric"), False),
    (("teacher_hypothetical", "native_energy_released_j"), 8.9),
    (("printed_answer", "ideal_thin_limit_j"), 0.),
])
def test_forged_solid_shell_provenance_charge_energy_and_shape_refused(path, value):
    forged = deepcopy(energy_record())
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        verify_energy(forged)


def test_no_misleading_time_or_fake_browser_computation(generated):
    _root, out, _manifest = generated
    for name in ("p1-external-work", "p3-solid-shell-energy"):
        text = (out/(name+".html")).read_text(encoding="utf-8")
        assert "<svg " in text
        assert "<script" not in text
        assert "TEACHER HYPOTHETICAL" in text
        assert "Original requested answer" in text
        assert "not cryptographically authenticated" in text
        assert "<output" in text
    p1 = (out/"p1-external-work.html").read_text(encoding="utf-8")
    assert "5-3-4" in p1 and "Work by external agent (+)" in p1
    assert "Work by electric field" in p1
    p3 = (out/"p3-solid-shell-energy.html").read_text(encoding="utf-8")
    assert p3.count('r="55"') == 2
    assert p3.count('r="110"') == 2
    assert "filled solid" in p3
    assert "two separately solved static equilibria" in p3
    assert "wire-current" in p3 and "equilibria" in p3
