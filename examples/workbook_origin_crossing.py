"""Workbook PDF page 14, Numerical Value Type Q6: speed at the origin."""
from math import hypot

from conductor_sim import FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec
from conductor_sim.motion_events import EventPolicy, EventStudy, PlaneEvent, investigate_motion
from conductor_sim.scenarios import PrescribedScenario, revise_scenario


def workbook_scenario():
    field = PrescribedField((FixedPointField(2e-6, (3., 0., 0.)),
                             FixedPointField(2e-6, (-3., 0., 0.))))
    spec = TestChargeMotionSpec(field, PointCharge(-4e-6, (0., 4., 0.)),
                                mass_kg=.001, initial_velocity_m_s=(0., 0., 0.),
                                duration_s=3., sample_interval_s=.05, plane_normal=(0., 0., 1.),
                                relative_tolerance=1e-9)
    return PrescribedScenario(
        "workbook-p14-q6", spec, (("q1", "Right fixed charge"), ("q2", "Left fixed charge")),
        "Workbook p14 numerical Q6, two equal fixed sources and release from rest. "
        "Vacuum Coulomb model. Integration horizon, plane and resolution are numerical choices.",
        origin="source-given")


def first_downward_crossing(scenario):
    detector = PlaneEvent("cross-y-zero", (0., 0., 0.), (0., 1., 0.),
                          "First downward crossing of y = 0", direction=-1, terminal=True)
    study = EventStudy(scenario, (detector,), EventPolicy(.01, 1e-9, 1e-8, 1e-7))
    run = investigate_motion(study)
    if len(run.events) != 1 or run.unresolved:
        raise ValueError("the first downward crossing was not resolved")
    return run.events[0].sample


def teacher_revision(original):
    return revise_scenario(original, scenario_id="teacher-unequal-sources",
                           source_changes={"q1": {"charge_c": 2.1e-6}},
                           note="Teacher increases only the right fixed charge to +2.1 microcoulombs.")


def main():
    original = workbook_scenario()
    at_origin = first_downward_crossing(original)
    if hypot(*at_origin.position_m) > 1e-8:
        raise ValueError("workbook crossing missed the origin")
    speed = hypot(*at_origin.velocity_m_s)
    print("Workbook p14 Q6: equal fixed charges cancel transverse force on the y-axis.")
    print(f"Speed at the origin: {speed:.6f} m/s; nearest integer {speed:.0f} m/s.")
    revised = first_downward_crossing(teacher_revision(original))
    print(f"Teacher changes q1 to +2.1 microcoulombs: first y=0 crossing at "
          f"x={revised.position_m[0]:+.6f} m; speed there {hypot(*revised.velocity_m_s):.6f} m/s.")


if __name__ == "__main__":
    main()
