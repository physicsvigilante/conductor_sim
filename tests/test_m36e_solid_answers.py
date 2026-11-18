"""M36e: original p26 visible answers are native, state-aware, and truthful.

No Playwright import here: generic Python matrix and source-native physics
remain independent of the DEDICATED real-Chromium gate.
"""
from __future__ import annotations

from copy import deepcopy
import re

import pytest

from conductor_sim.viewer import TEMPLATE
from tools.m36d_true_solid_sphere import (
    _source_answer_panel, run_all,
)
from tools.m36d_solid_reference import check as independent_si


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    output=tmp_path_factory.mktemp("m36e-true-solid-p26-source-answers")
    report=run_all(output)
    return output,report


def test_true_source_physics_unchanged_and_original_case_count_one(native):
    _folder,report=native
    si=independent_si(report)
    assert si["qualified_original_jee_solid_sphere_apparatus"]==1
    assert si["teacher_real_physical_revisions"]==2
    assert si["actual_solid_source_states_independently_checked"]==3
    assert report["original_source_apparatus_count"]==1
    assert report["physical_time_evolution"] is False
    assert report["native_states"][0]["case_role"].startswith("source_original_shape")
    assert report["native_states"][1]["case_role"].startswith("teacher_")


def test_source_specific_readout_panel_has_three_native_physical_states(native):
    directory,report=native
    text=(directory/report["viewer_file"]).read_text(encoding="utf-8")
    assert '<section id="m36e-source-observations"' in text
    assert '<script id="m36e-static-answer-synchronizer">' in text
    assert 'frameInput.addEventListener("input",update)' in text
    assert 'panel.hidden=Number(panel.dataset.m36eState)!==index' in text
    assert "The printed source supplies SYMBOLIC R and Q" in text
    assert "NOT a time trajectory" in text
    assert "Surface-charge density (one real outer face)" in text
    assert 'aria-label="Surface-charge density along a great circle"' in text
    assert "Induced surface-charge profile" not in text
    assert "Induced charge density along a great circle" not in text
    # The general legacy viewer is unchanged; only this one source-specific
    # physical world gets its honest isolated-net-Q label.
    assert "Induced surface-charge profile" in TEMPLATE
    assert "Surface-charge density (one real outer face)" not in TEMPLATE
    assert 'data-m36e-state="0"' in text
    assert 'data-m36e-state="1" hidden' in text
    assert 'data-m36e-state="2" hidden' in text


def test_each_original_or_teacher_readout_is_actually_from_saved_native_observables(native):
    folder,report=native
    text=(folder/report["viewer_file"]).read_text(encoding="utf-8")
    cards=re.findall(
        r'<section data-m36e-state="([012])" data-m36e-role="([^"]+)"'
        r'(?: hidden)?>(.*?)</section>',
        text,re.DOTALL,
    )
    assert len(cards)==3
    assert [i for i,_,_ in cards]==["0","1","2"]
    for i,role,card in cards:
        actual=report["native_states"][int(i)]
        assert role==actual["case_role"]
        assert ("PRINTED ORIGINAL" in card if i=="0" else "TEACHER HYPOTHETICAL" in card)
        native=actual["native_samples"]
        expected={
            "V_half_R":(native["half_radius_potential_v"],"V"),
            "V_three_half_R":(native["three_half_radius_potential_v"],"V"),
            "E_inside":(
                sum(v*v for v in native["metal_half_radius_e_v_m"])**.5,"V/m"
            ),
            "sigma_outer":(native["outer_sigma_c_m2"],"C/m²"),
        }
        readings=re.findall(
            r'<strong data-m36e-number="([^"]+)" '
            r'data-native-value="([^"]+)">([^<]+)</strong>',
            card,
        )
        assert len(readings)==4
        assert set(key for key,_,_ in readings)==set(expected)
        for key,stored,visible in readings:
            physical,unit=expected[key]
            assert float(stored)==pytest.approx(physical,rel=2e-12,abs=1e-19)
            assert visible.endswith(" "+unit)
            assert float(visible.split()[0])==pytest.approx(
                physical,rel=2e-9,abs=1e-13,
            )
        assert "Only one physical OUTER charge-density face exists" in card
    assert "V(R/2)" in cards[0][2] and "V(3R/2)" in cards[0][2]


def test_source_answer_panel_reacts_to_native_values_not_prefilled_gauss_formula(native):
    _folder,report=native
    states=deepcopy(report["native_states"])
    # Deliberately wrong saved native sample changes the visible answer card:
    # separate Gauss reference MUST reject the physical forgery.
    old_value=states[0]["native_samples"]["half_radius_potential_v"]
    states[0]["native_samples"]["half_radius_potential_v"]=123456.789
    html=_source_answer_panel(states)
    assert 'data-native-value="123456.789"' in html
    assert f'data-native-value="{format(old_value,".17g")}"' not in html
    forged=deepcopy(report)
    forged["native_states"][0]["native_samples"]["half_radius_potential_v"]=123456.789
    with pytest.raises(AssertionError,match="Gauss"):
        independent_si(forged)


def test_html_escapes_teacher_role_in_source_readout_attribute(native):
    _folder,report=native
    states=deepcopy(report["native_states"])
    malicious='" onmouseover="evil()" <script>alert(1)</script>'
    states[1]["case_role"]=malicious
    generated=_source_answer_panel(states)
    assert 'onmouseover="evil()"' not in generated
    assert '<script>alert(1)</script>' not in generated
    assert "&quot;" in generated and "&lt;script&gt;" in generated


def test_real_rendered_field_grid_is_denser_than_previous_37(native):
    folder,report=native
    html=(folder/report["viewer_file"]).read_text(encoding="utf-8")
    import json
    match=re.search(
        r'<script id="simulation-data" type="application/json">(.*?)</script>',
        html,re.DOTALL,
    )
    assert match
    bundle=json.loads(match.group(1))
    assert bundle["schema"]=="conductor_sim.visualization.v1"
    assert len(bundle["frames"])==3
    for frame in bundle["frames"]:
        assert len(frame["grid"]["x_m"])==65
        assert len(frame["grid"]["y_m"])==65
        assert len(frame["grid"]["region"])==65
        assert all(len(row)==65 for row in frame["grid"]["region"])
        assert len(frame["geometry"])==1
        assert frame["geometry"][0]["inner_radius_m"]==0
        assert [p["surface"] for p in frame["profiles"]]==["outer"]
    assert report["actual_rendered_geometry"][0]["inner_radius_m"]==0.
