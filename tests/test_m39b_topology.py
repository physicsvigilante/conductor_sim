"""M39b: true Gauss ledger identity --> honest static topology-only display."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from conductor_sim.cavity_gauss import (
    CavityCharge, GaussCavityAssembly, SealedCavity,
    solve_cavity_gauss,
)
from conductor_sim.cavity_topology_viewer import (
    _si, topology_view_manifest, render_cavity_topology,
    WARNING, PROVENANCE, REFUSAL,
)
from conductor_sim.model import FixedNetCharge, Grounded
from conductor_sim.solver import UnsupportedProblem
from tools.m39a_gauss_example import sample_assembly
from tools.m39b_topology_example import run_all
from tools.m39b_topology_reference import check_topology_view_manifest


def pair(spec):
    return (
        solve_cavity_gauss(spec),
        solve_cavity_gauss(replace(spec, constraint=Grounded())),
    )


def test_genuine_teacher_example_and_real_standalone_oracle(tmp_path):
    manifest = run_all(tmp_path)
    result = check_topology_view_manifest(manifest)
    assert result["cavities_checked"] == 3
    assert result["source_identities_checked"] == 4
    assert result["original_workbook_items_newly_solved"] == 0
    for case in ("isolated", "grounded"):
        saved = json.loads((tmp_path / ("m39b-" + case + "-native-gauss.json")).read_text())
        assert saved == manifest["states"][case]
    check = subprocess.run(
        [sys.executable, "-m", "tools.m39b_topology_reference",
         "--input", str(tmp_path / "m39b-topology-view.json")],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(check.stdout)["cavities_checked"] == 3
    assert (tmp_path / "m39b-topology-view.html").read_text(encoding="utf-8") == (
        render_cavity_topology(*pair(sample_assembly()))
    )


@pytest.mark.parametrize("n", range(1, 9))
def test_composable_one_to_eight_cavities_neutral_and_signed_face_integrals(n):
    spec = GaussCavityAssembly("Metal", tuple(
        SealedCavity(f"Void_{j}", (
            CavityCharge(f"source_{j}_plus", (j + 1) * 1e-9),
            CavityCharge(f"source_{j}_minus", -(j + 1) * 1e-9),
        )) for j in range(n)), FixedNetCharge(1.25e-9))
    a, b = pair(spec)
    m = topology_view_manifest(a, b)
    assert len(m["cavities"]) == n
    assert all(row["source_charge_sum_c"] == 0.0 for row in m["cavities"])
    assert all(row["induced_inner_face_total_charge_c"] == 0.0 for row in m["cavities"])
    assert m["states"]["isolated"]["outer_face_total_charge_c"] == 1.25e-9
    assert m["states"]["grounded"]["outer_face_total_charge_c"] == 0
    check_topology_view_manifest(m)
    html = render_cavity_topology(a, b)
    assert html.count('class="cavity"') == n
    assert html.count("signed cancellation (local sigma unknown)") == n
    assert "TOPOLOGY ONLY" in html
    assert "Not generally implied for isolated metal" in html
    assert "<script" not in html.lower()


def test_empty_different_from_signed_neutral_with_correct_si():
    a, b = pair(GaussCavityAssembly("Metal", (
        SealedCavity("Empty"),
        SealedCavity("Balanced", (
            CavityCharge("P", 3e-9), CavityCharge("N", -3e-9),
        )),
        SealedCavity("Nonzero", (CavityCharge("NEG", -1e-9),)),
    ), FixedNetCharge(1e-9)))
    page = render_cavity_topology(a, b)
    assert 'id="m39b-source-empty-0"' in page
    assert "signed cancellation (local sigma unknown)" in page
    assert 'id="m39b-q-1">0 C</strong>' in page
    assert 'id="m39b-inner-1">0 C</strong>' in page
    assert "−1e-09 C" in page or "−1e-09" in page
    assert 'id="m39b-theorem-grounded"' in page
    assert 'id="m39b-original-count"' in page
    assert "Card positions, sizes and ordering" in page
    assert check_topology_view_manifest(topology_view_manifest(a, b))["cavities_checked"] == 3


def test_permuted_source_order_and_teacher_parent_immutable():
    spec = sample_assembly()
    saved_parent = solve_cavity_gauss(spec).as_dict()
    reordered = replace(spec.cavities[0], sources=tuple(reversed(spec.cavities[0].sources)))
    revised = replace(spec, cavities=(reordered,) + spec.cavities[1:])
    a, b = pair(revised)
    manifest = topology_view_manifest(a, b)
    assert [s["source_id"] for s in manifest["cavities"][0]["authored_sources"]] == [
        "source_A_negative", "source_A_positive",
    ]
    assert solve_cavity_gauss(spec).as_dict() == saved_parent
    check_topology_view_manifest(manifest)


@pytest.mark.parametrize("mode", [
    "grounded-first", "isolated-second", "different-metal", "different-epsilon",
    "different-source", "different-source-identity", "different-assignment",
])
def test_bad_pair_is_refused(mode):
    base = sample_assembly()
    a, b = pair(base)
    if mode == "grounded-first":
        args = (b, a)
    elif mode == "isolated-second":
        args = (a, a)
    elif mode == "different-metal":
        args = (a, solve_cavity_gauss(replace(base, conductor_id="Metal_B",constraint=Grounded())))
    elif mode == "different-epsilon":
        args = (a, solve_cavity_gauss(replace(base,permittivity_f_per_m=1.0,constraint=Grounded())))
    elif mode in ("different-source", "different-source-identity"):
        cav = base.cavities[0]
        source = cav.sources[0]
        amended = replace(source, charge_c=source.charge_c + 1e-9) if mode == "different-source" else (
            replace(source, source_id="renamed")
        )
        changed = replace(base, cavities=(
            replace(cav, sources=(amended,) + cav.sources[1:]), *base.cavities[1:]
        ), constraint=Grounded())
        args = (a, solve_cavity_gauss(changed))
    else:
        cav1, cav2, cav3 = base.cavities
        moved = cav2.sources[0]
        changed = replace(base, constraint=Grounded(), cavities=(
            cav1, replace(cav2, sources=()), replace(cav3, sources=cav3.sources + (moved,)),
        ))
        args = (a, solve_cavity_gauss(changed))
    with pytest.raises((ValueError,TypeError)):
        topology_view_manifest(*args)


def test_tampered_native_ledger_is_not_renderable():
    a, b = pair(sample_assembly())
    with pytest.raises(ValueError):
        render_cavity_topology(replace(a, outer_face_charge_c=False), b)
    face = replace(b.faces[0], induced_inner_face_charge_c=100.0)
    with pytest.raises(ValueError):
        topology_view_manifest(a, replace(b, faces=(face,) + b.faces[1:]))


@pytest.mark.parametrize("key,bad", [
    ("source_role", "original_jee"),
    ("original_workbook_items_newly_solved", True),
    ("geometry_verified", True),
    ("diagram_is_spatial_geometry", True),
    ("time_evolution_defined", True),
    ("warning", "This is the real cavity geometry"),
    ("unsupported", "we solved the local electric field"),
])
def test_hostile_manifest_claims_are_refused(key,bad):
    a, b = pair(sample_assembly())
    forged = deepcopy(topology_view_manifest(a,b))
    forged[key] = bad
    with pytest.raises(ValueError):
        check_topology_view_manifest(forged)


@pytest.mark.parametrize("field,bad", [
    ("induced_inner_face_total_charge_c", False),
    ("source_charge_sum_c", False),
    ("source_charge_sum_c", 9e-9),
    ("cavity_id", "fake_cavity"),
])
def test_hostile_visible_face_values_and_id_are_refused(field,bad):
    a, b = pair(sample_assembly())
    forged = deepcopy(topology_view_manifest(a,b))
    forged["cavities"][0][field] = bad
    with pytest.raises(ValueError):
        check_topology_view_manifest(forged)


@pytest.mark.parametrize("path,bad", [
    (("states","isolated","outer_face_total_charge_c"),False),
    (("states","grounded","outer_face_total_charge_c"),1e-12),
    (("states","isolated","conductor","constraint"),"grounded_zero_potential"),
    (("states","grounded","units","charge"),"nC"),
    (("states","isolated","cavities",0,"authored_sources",0,"charge_c"),True),
])
def test_hostile_source_si_and_state_claims_refused(path,bad):
    m = deepcopy(topology_view_manifest(*pair(sample_assembly())))
    v = m
    for p in path[:-1]:
        v = v[p]
    v[path[-1]] = bad
    with pytest.raises(ValueError):
        check_topology_view_manifest(m)


def test_html_is_deterministic_static_escaped_and_never_external():
    a,b = pair(sample_assembly())
    html1 = render_cavity_topology(a,b)
    html2 = render_cavity_topology(a,b)
    assert html1 == html2
    assert html1.startswith("<!doctype html>")
    assert "<script" not in html1.lower()
    assert "<iframe" not in html1.lower()
    assert "http://" not in html1 and "https://" not in html1
    for text in (WARNING, PROVENANCE, REFUSAL, "TEACHER HYPOTHETICAL",
                 "NOT time evolution", "original JEE"):
        assert text.lower() in html1.lower()
    with pytest.raises(ValueError):
        GaussCavityAssembly("Metal", (SealedCavity("Q<script>"),), Grounded())


def test_smallest_subnormal_is_not_misrepresented_as_visual_zero():
    tiny = 5e-324
    spec = GaussCavityAssembly("Metal", (
        SealedCavity("Tiny", (CavityCharge("qTiny", tiny),)),
    ), FixedNetCharge(0.0), 1.0)
    a,b = pair(spec)
    m = topology_view_manifest(a,b)
    check_topology_view_manifest(m)
    page = render_cavity_topology(a,b)
    assert "+5e-324 C" in page
    assert "−5e-324 C" in page
    assert 'id="m39b-outer-isolated">+5e-324 C</td>' in page
    assert "0 C" in page
    assert _si(tiny, "C") != _si(0.0, "C")


def test_nonrepresentable_flux_is_refused_instead_of_becoming_zero():
    spec = GaussCavityAssembly("Metal", (SealedCavity("Empty"),),
                               FixedNetCharge(1e-300),1e100)
    with pytest.raises(UnsupportedProblem):
        solve_cavity_gauss(spec)


def test_markup_escape_for_ids_and_required_provenance_anchored():
    a,b = pair(sample_assembly())
    html = render_cavity_topology(a,b)
    assert 'id="m39b-metal"' in html
    assert 'id="m39b-membership"' in html
    assert 'id="m39b-refusal"' in html
    assert 'id="m39b-original-count"' in html
    assert 'id="m39b-summary"' in html
    assert 'id="m39b-cavities"' in html
    assert "<ul>" in html and "</ul>" in html
