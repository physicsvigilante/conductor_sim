"""Command contract tests using a fake API. These are NOT numerical FEMM tests."""
import importlib.util
from math import pi, sqrt
from pathlib import Path

import pytest

from conductor_sim import ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell
from conductor_sim.femm5 import (
    Femm5Backend, Femm5ExecutionError, Femm5Session, Femm5Settings,
    Femm5Unavailable, _SOURCE_CORE, _load_binding,
)
from conductor_sim.solver import SingularField, UnsupportedProblem


class FakeClient:
    def __init__(self, fail_on=None, point_data=None):
        self.calls = []
        self.fail_on = fail_on
        self.point_data = point_data or (12, 0, 0, 4, 6, 1, 1, 0)

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            if name == self.fail_on:
                raise RuntimeError("synthetic FEMM fault")
            if name == "eo_getconductorproperties":
                return (35.0, 1e-9) if args[0] == _SOURCE_CORE else (12.0, 2e-9)
            if name == "eo_getpointvalues":
                return self.point_data
            return None
        return call


class FakeSession:
    def __init__(self, client):
        self.client = client
        self.opens = 0
        self.closes = 0

    def open(self):
        self.opens += 1
        return self.client

    def close(self):
        self.closes += 1


def problem(charge=2e-9, center=(0.0, 0.0, 0.0)):
    return ElectrostaticProblem(
        conductors=(SphericalShell(.1, .12, FixedNetCharge(charge), center_m=center),)
    )


def make_backend(fail_on=None):
    fake = FakeClient(fail_on=fail_on)
    sess = FakeSession(fake)
    return Femm5Backend(session_factory=lambda: sess), fake, sess


def calls(fake, name):
    return [args for cmd, args in fake.calls if cmd == name]


def test_empty_shell_compilation_and_conductor_identity():
    backend, fake, session = make_backend()
    p = problem()
    with backend.solve(p) as solution:
        assert backend.supports(p)
        assert solution.conductor_results[0].potential_v == 12
        assert solution.conductor_results[0].net_charge_c == 2e-9
        assert session.closes == 0
    assert session.opens == session.closes == 1
    assert calls(fake, "newdocument") == [(1,)]
    assert calls(fake, "ei_probdef") == [("meters", "axi", 1e-9)]
    assert calls(fake, "ei_addconductorprop") == [("shell", 0, 2e-9, 0)]
    assert len(calls(fake, "ei_addarc")) == 6  # two arcs per each of three radii
    assert len(calls(fake, "ei_addsegment")) == 5  # axis subdivided at every interface
    assert len(calls(fake, "ei_addblocklabel")) == 3  # cavity, metal, exterior
    assert calls(fake, "ei_analyze") == [(1,)]
    assert calls(fake, "ei_loadsolution") == [()]
    assert calls(fake, "ei_saveas")[0][0].endswith("charged_shell.fee")
    assert not Path(calls(fake, "ei_saveas")[0][0]).parent.exists()
    # Both inner/outer surfaces have the SAME conductor name; outer open boundary does not.
    props = calls(fake, "ei_setarcsegmentprop")
    assert len(props) == 6
    assert len([a for a in props if a[4] == "shell"]) == 4
    assert len([a for a in props if a[1] == "open_robin" and a[4] == "<None>"]) == 2
    # Open boundary is Robin (format=1), coefficient epsilon/R (not imposed voltage).
    args = calls(fake, "ei_addboundprop")[0]
    assert args[:3] == ("open_robin", 0, 0)
    assert args[3] > 0 and args[4:] == (0, 1)


def test_reproducible_coordinate_transform_and_live_queries():
    backend, fake, session = make_backend()
    p = problem(center=(3.0, -4.0, 2.0))
    with backend.solve(p) as solution:
        assert solution.potential_at((3.03, -3.96, 2.07)) == 12
        ex, ey, ez = solution.electric_field_at((3.03, -3.96, 2.07))
        assert (ex, ey, ez) == pytest.approx((2.4, 3.2, 6))
        assert solution.electric_field_at((3, -4, 2.05)) == pytest.approx((0, 0, 6))
        with pytest.raises(SingularField):
            solution.electric_field_at((3.12, -4, 2.0))
        with pytest.raises(UnsupportedProblem):
            solution.potential_at((10, -4, 2.0))
    assert calls(fake, "eo_getpointvalues")[0] == pytest.approx((.05, .07))
    with pytest.raises(RuntimeError):
        solution.potential_at((3, -4, 2))
    assert session.closes == 1


def test_cleanup_on_solve_error():
    backend, fake, session = make_backend(fail_on="ei_analyze")
    with pytest.raises(Femm5ExecutionError, match="synthetic FEMM fault"):
        backend.solve(problem())
    assert session.closes == 1
    path = Path(calls(fake, "ei_saveas")[0][0])
    assert not path.parent.exists()


def test_bad_output_does_not_masquerade_as_success():
    backend, fake, session = make_backend()
    def reject(name):
        return ("NaN", 2e-9)
    fake.eo_getconductorproperties = reject
    with pytest.raises(Femm5ExecutionError, match="nonfinite"):
        backend.solve(problem())
    assert session.closes == 1


def test_missing_femm_binding_is_explicit(monkeypatch):
    monkeypatch.delenv("FEMM5_BINDING", raising=False)
    with pytest.raises(Femm5Unavailable, match="FEMM5_BINDING"):
        _load_binding(None)
    with pytest.raises(Femm5Unavailable, match="existing femm.py"):
        _load_binding("/not/the/femm5/binding/femm.py")


def test_binding_rejects_femm42_lookalike(tmp_path):
    binding = tmp_path / "femm.py"
    binding.write_text("def openfemm(x): pass\n", encoding="utf-8")
    with pytest.raises(Femm5Unavailable, match="FemmSession"):
        _load_binding(binding)


def test_femm5_session_rejects_bad_port():
    with pytest.raises(ValueError, match="port"):
        Femm5Session(port=5)


def test_strict_capability_boundary():
    backend, _, session = make_backend()
    assert backend.supports(problem())
    bad = [
        ElectrostaticProblem(),
        ElectrostaticProblem(charges=(PointCharge(1e-9, (.103, 0, 0)),), conductors=problem().conductors),
        ElectrostaticProblem(charges=(PointCharge(1e-9, (.03, 0, 0)), PointCharge(2e-9, (-.03, 0, 0))), conductors=problem().conductors),
        ElectrostaticProblem(charges=(PointCharge(1e-9),), conductors=(SphericalShell(.1,.12,Grounded(),name=_SOURCE_CORE),)),
        ElectrostaticProblem(conductors=(
            SphericalShell(.1,.12,FixedNetCharge(0),name="a"),
            SphericalShell(.2,.3,FixedNetCharge(0),name="b"),
        )),
    ]
    for p in bad:
        assert not backend.supports(p)
        with pytest.raises(UnsupportedProblem):
            backend.solve(p)
    assert session.opens == 0


@pytest.mark.parametrize("field,value", [
    ("outer_radius_factor", 2),
    ("arc_degrees", 20),
    ("mesh_scale", 0),
    ("solver_precision", 1),
    ("source_radius_fraction", .01),
    ("source_radius_fraction", .21),
    ("source_radius_fraction", float("nan")),
])
def test_invalid_settings(field, value):
    with pytest.raises(ValueError):
        Femm5Settings(**{field: value})


def test_settings_change_mesh_and_outer_boundary():
    fake = FakeClient()
    session = FakeSession(fake)
    settings = Femm5Settings(outer_radius_factor=12, arc_degrees=2.5, mesh_scale=.5)
    b = Femm5Backend(settings=settings, session_factory=lambda: session)
    with b.solve(problem()) as sol:
        assert sol.outer_radius_m == pytest.approx(.12 * 12)
        assert sol.potential_at((.5, 0, 0)) == 12
    assert calls(fake, "ei_addboundprop")[0][3] == pytest.approx(8.8541878128e-12 / (12*.12))
    assert all(a[0] == 2.5 for a in calls(fake, "ei_setarcsegmentprop"))


def test_close_is_idempotent():
    backend, fake, session = make_backend()
    sol = backend.solve(problem())
    sol.close()
    sol.close()
    assert session.closes == 1


def test_grounded_empty_shell_is_fixed_zero_voltage():
    backend, fake, session = make_backend()
    p = ElectrostaticProblem(conductors=(SphericalShell(.1, .12, Grounded()),))
    with backend.solve(p) as sol:
        assert sol.source_radius_m is None
        assert sol.numerical_source_charge_c is None
        assert len(sol.conductor_results) == 1
    assert session.opens == session.closes == 1
    assert calls(fake, "ei_addconductorprop") == [("shell", 0, 0, 1)]
    assert len(calls(fake, "ei_addarc")) == 6


def test_centered_charge_uses_separate_charged_sphere_not_pointprop():
    backend, fake, session = make_backend()
    p = ElectrostaticProblem(
        charges=(PointCharge(1e-9),),
        conductors=(SphericalShell(.1, .12, FixedNetCharge(0)),),
    )
    with backend.solve(p) as sol:
        assert sol.source_radius_m == pytest.approx(.01)
        assert sol.numerical_source_charge_c == pytest.approx(1e-9)
        assert sol.conductor_results[0].inner_surface_charge_c == pytest.approx(-1e-9)
        assert sol.conductor_results[0].outer_surface_charge_c == pytest.approx(3e-9)
        with pytest.raises(UnsupportedProblem, match="regularized source core"):
            sol.potential_at((.005, 0, 0))
        with pytest.raises(UnsupportedProblem, match="regularized source core"):
            sol.potential_at((0, 0, 0))
        with pytest.raises(SingularField, match="jump"):
            sol.electric_field_at((.01, 0, 0))
        assert sol.potential_at((.03, 0, 0)) == 12
    assert session.opens == session.closes == 1
    assert calls(fake, "ei_addconductorprop") == [
        ("shell", 0, 0, 0), (_SOURCE_CORE, 0, 1e-9, 0),
    ]
    assert len(calls(fake, "ei_addarc")) == 8
    assert len(calls(fake, "ei_addsegment")) == 7
    assert len(calls(fake, "ei_addblocklabel")) == 4
    assert len([a for a in calls(fake,"ei_setarcsegmentprop") if a[4]=="shell"]) == 4
    assert len([a for a in calls(fake,"ei_setarcsegmentprop") if a[4]==_SOURCE_CORE]) == 2
    assert calls(fake, "ei_addpointprop") == []


def test_grounded_and_centered_charge_preserves_one_physical_conductor():
    backend, fake, _ = make_backend()
    q = -2e-9
    p = ElectrostaticProblem(
        charges=(PointCharge(q, (1, 2, 3)),),
        conductors=(SphericalShell(.2, .25, Grounded(), center_m=(1, 2, 3), name="metal"),),
    )
    with backend.solve(p) as sol:
        assert sol.source_radius_m == pytest.approx(.02)
        assert len(sol.conductor_results) == 1
        assert sol.numerical_source_charge_c == pytest.approx(1e-9) # fake FEMM result
        assert sol.potential_at((1.04, 2, 3)) == 12
    assert calls(fake, "ei_addconductorprop") == [
        ("metal", 0, 0, 1), (_SOURCE_CORE, 0, q, 0)
    ]


def test_centered_charges_compose_as_total_source_charge():
    backend, fake, _ = make_backend()
    p = ElectrostaticProblem(
        charges=(PointCharge(2e-9), PointCharge(-1e-9)),
        conductors=problem().conductors,
    )
    with backend.solve(p):
        pass
    assert calls(fake, "ei_addconductorprop")[1] == (_SOURCE_CORE, 0, 1e-9, 0)


def test_cancelling_centered_charges_do_not_create_regularization_core():
    backend, fake, _ = make_backend()
    p = ElectrostaticProblem(
        charges=(PointCharge(2e-9), PointCharge(-2e-9)),
        conductors=problem().conductors,
    )
    with backend.solve(p) as sol:
        assert sol.source_radius_m is None
        assert sol.potential_at((0, 0, 0)) == 12
    assert len(calls(fake, "ei_addarc")) == 6
    assert len(calls(fake, "ei_addconductorprop")) == 1


def test_source_radius_is_configurable():
    fake, session = FakeClient(), None
    session = FakeSession(fake)
    p = ElectrostaticProblem(charges=(PointCharge(1e-9),), conductors=problem().conductors)
    b = Femm5Backend(settings=Femm5Settings(source_radius_fraction=.05), session_factory=lambda: session)
    with b.solve(p) as sol:
        assert sol.source_radius_m == pytest.approx(.005)
    assert any(args[0] == pytest.approx(.005) and args[1] == 0 for args in calls(fake, "ei_addarc"))


def test_source_charge_bad_output_cleans_session():
    backend, fake, session = make_backend()
    def reject(name):
        return (12., 2e-9) if name != _SOURCE_CORE else ("NaN", 1e-9)
    fake.eo_getconductorproperties = reject
    p = ElectrostaticProblem(charges=(PointCharge(1e-9),), conductors=problem().conductors)
    with pytest.raises(Femm5ExecutionError, match="nonfinite"):
        backend.solve(p)
    assert session.closes == 1
