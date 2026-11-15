"""Post-M22c: reproduce actual observation and saved-data failures.

No new source laws, dynamics, symmetry admissions or numerical policy.
"""
from copy import deepcopy
from math import pi
import xml.etree.ElementTree as ET

import pytest

from conductor_sim import (EPSILON_0, FixedPointField, PrescribedField,
    UniformDiskAxisField, UniformElectricField, UniformRingAxisField,
    UniformRodField, UniformShellField, UniformSphereField)
from conductor_sim.jee_potential_explorer import (
    endpoint_field_work, render_static_profile, sample_fixed_potential_map,
    sample_static_profile, validate_fixed_potential_map)
from conductor_sim.prescribed_viewer_validation import validate_prescribed_viewer
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.visualization import SliceSpec
from examples.mirrored_rods import create_case

K = 1 / (4 * pi * EPSILON_0)


@pytest.fixture(scope="module")
def mirrored():
    return create_case("P1")[1]["viewer"]


@pytest.mark.parametrize("indices", [[False, True], [False, 1], [0, True],
                                     [0., 1.], ["0", "1"], {"0": 0, "1": 1}])
def test_omission_indices_require_actual_integers(mirrored, indices):
    bad = deepcopy(mirrored)
    bad["prescribed_motion"]["mirrored_rod_source_indices"] = indices
    with pytest.raises(ValueError, match="off-plane rod"):
        validate_prescribed_viewer(bad)


def test_real_integer_omissions_unchanged(mirrored):
    assert validate_prescribed_viewer(mirrored)
    assert mirrored["prescribed_motion"]["mirrored_rod_source_indices"] == [0, 1]


@pytest.fixture(scope="module")
def static_map():
    return sample_fixed_potential_map(
        PrescribedField((UniformElectricField((1., 0., 0.)),)),
        SliceSpec(origin_m=(0., 0., 0.), u=(1., 0., 0.), v=(0., 1., 0.), resolution=9),
        source_annotations=(("field:test", "Applied field"),),
        source_mask_m=.001, line_steps=8)


@pytest.mark.parametrize("vector", [[], [1.], [1., 0.], [1., 0., 0., 0.],
    [True, 0., 0.], [1., False, 0.], [1., 0., float("nan")],
    [1., 0., float("inf")], [1., 0., "0"], {"x": 1., "y": 0., "z": 0.}])
def test_static_map_rejects_malformed_vectors(static_map, vector):
    bad = deepcopy(static_map)
    bad["frames"][0]["grid"]["electric_field_v_per_m"][0][0] = vector
    with pytest.raises(ValueError):
        validate_fixed_potential_map(bad)


@pytest.mark.parametrize("value", [True, False, "0", float("nan"), float("inf")])
def test_static_map_rejects_nonphysical_scalar_types(static_map, value):
    bad = deepcopy(static_map)
    # The centre node is exactly V=0, so False previously passed as a number.
    bad["frames"][0]["grid"]["potential_v"][4][4] = value
    with pytest.raises(ValueError):
        validate_fixed_potential_map(bad)


def test_static_map_region_row_width_checked(static_map):
    bad = deepcopy(static_map)
    bad["frames"][0]["grid"]["region"][0] = []
    with pytest.raises(ValueError):
        validate_fixed_potential_map(bad)


def labels(field):
    return tuple((f"source:{i}", f"Source {i}") for i, _ in enumerate(field.sources))


def work(field, start, end):
    return endpoint_field_work(field, source_annotations=labels(field),
                               probe_charge_c=1e-9, start_m=start, end_m=end)


@pytest.mark.parametrize("signed_q", [1e-9, -1e-9])
def test_shell_surface_work_requires_potential_not_defined_surface_field(signed_q):
    field = PrescribedField((UniformShellField(signed_q, .101),))
    result = work(field, (.101, 0., 0.), (.2, 0., 0.))
    assert result["field_work_j"] == pytest.approx(1e-9 * K * signed_q * (1/.101 - 1/.2))
    assert result["start_electric_field_v_m"] is None
    assert result["end_electric_field_v_m"] == pytest.approx([K*signed_q/.2**2, 0., 0.])
    assert "start" in result["unavailable_endpoint_fields"]
    assert "undefined" in result["caveat"]


def test_disk_surface_work_and_superposed_field():
    field = PrescribedField((UniformDiskAxisField(2e-9, .08), UniformElectricField((0., 0., 7.))))
    result = work(field, (0., 0., 0.), (0., 0., .15))
    h = (.08**2 + .15**2)**.5
    expected = 1e-9 * (2*K*2e-9/.08 - 2*K*2e-9/(h+.15) + 7*.15)
    assert result["field_work_j"] == pytest.approx(expected)
    assert result["start_electric_field_v_m"] is None
    assert result["source_work_components"][1]["field_work_j"] == pytest.approx(7*.15e-9)
    reverse = work(field, (0., 0., .15), (0., 0., 0.))
    assert reverse["field_work_j"] == pytest.approx(-expected)
    assert reverse["end_electric_field_v_m"] is None


def test_both_endpoint_fields_can_be_undefined_without_losing_work():
    field = PrescribedField((UniformShellField(1e-9, .1), UniformShellField(-2e-9, .2)))
    result = work(field, (.1, 0., 0.), (.2, 0., 0.))
    assert result["field_work_j"] == pytest.approx(1e-9*K*1e-9*(10-5))
    assert result["start_electric_field_v_m"] is None
    assert result["end_electric_field_v_m"] is None
    assert set(result["unavailable_endpoint_fields"]) == {"start", "end"}


def test_singular_potential_and_unsupported_field_law_still_refused():
    with pytest.raises(UnsupportedProblem, match="singularity"):
        work(PrescribedField((FixedPointField(1e-9),)), (0., 0., 0.), (.2, 0., 0.))
    with pytest.raises(UnsupportedProblem, match="axis"):
        work(PrescribedField((UniformRingAxisField(1e-9, .1),)), (.1, 0., .1), (0., 0., .2))


def profile(field, *, origin=(0., 0., 0.), axis=(1., 0., 0.), start=0., end=.24, coordinate="axial"):
    return sample_static_profile(field, source_annotations=labels(field), origin_m=origin,
        axis=axis, start_m=start, end_m=end, sample_count=21, coordinate=coordinate)


def polylines(record):
    html = render_static_profile(record, title="Regression audit", source_note="Teacher hypothetical")
    # Only inspect the two actual profile SVGs; do not infer correctness from pixels.
    fragments = ["<svg" + frag.split("</svg>")[0] + "</svg>" for frag in html.split("<svg")[1:]]
    return [len(ET.fromstring(f).findall("{http://www.w3.org/2000/svg}polyline")) for f in fragments]


def test_unsampled_shell_jump_does_not_become_sloping_e_curve():
    p = profile(PrescribedField((UniformShellField(1e-9, .101),)), coordinate="radial")
    assert len(p["rows"]) == 21
    assert not any(row["field_undefined"] for row in p["rows"])
    breaks = [r for r in p["rows"] if r.get("break_before")]
    assert len(breaks) == 1
    assert "field_parallel_v_m" in breaks[0]["break_before"]
    assert "potential_v" not in breaks[0]["break_before"]
    assert polylines(p) == [2, 1]
    for r in p["rows"]:
        x = r["coordinate_m"]
        assert r["potential_v"] == pytest.approx(K*1e-9/max(.101, x))
        assert r["field_parallel_v_m"] == pytest.approx(0. if x < .101 else K*1e-9/x**2)


def test_unsampled_disc_jump_preserves_continuous_potential():
    p = profile(PrescribedField((UniformDiskAxisField(1e-9, .1),)),
                axis=(0., 0., 1.), start=-.13, end=.17)
    assert not any(r["field_undefined"] for r in p["rows"])
    assert polylines(p) == [2, 1]


def test_two_shell_crossings_between_outside_samples_still_break():
    p = profile(PrescribedField((UniformShellField(1e-9, .002, (.102, 0., 0.)),)))
    assert all(abs(r["coordinate_m"]-.102) > .002 for r in p["rows"])
    assert polylines(p) == [2, 1]


@pytest.mark.parametrize("source", [FixedPointField(1e-9, (.101, 0., 0.)),
    UniformRodField(1e-9, (.101, 0., -.03), (.101, 0., .03))])
def test_unsampled_poles_break_both_potential_and_field(source):
    p = profile(PrescribedField((source,)))
    assert all(r["potential_v"] is not None for r in p["rows"])
    assert polylines(p) == [2, 2]


def test_smooth_source_combinations_do_not_gain_artificial_gaps():
    p = profile(PrescribedField((UniformSphereField(1e-9, .101),
                                UniformRingAxisField(-.5e-9, .1, axis=(1., 0., 0.)),
                                UniformElectricField((2., 0., 0.)))))
    assert not any(r.get("break_before") for r in p["rows"])
    assert polylines(p) == [1, 1]


def test_already_sampled_surface_needs_no_new_gap_metadata():
    p = profile(PrescribedField((UniformShellField(1e-9, .12),)), coordinate="radial")
    assert sum(r["field_undefined"] for r in p["rows"]) == 1
    assert not any(r.get("break_before") for r in p["rows"])
    assert polylines(p) == [2, 1]
