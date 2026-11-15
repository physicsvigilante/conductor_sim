"""M26a public-API example: structurally add a held source to driven motion.

No per-case solver; field, drag, schedule and numerical investigations
remain independently authored. Held charges do not respond to the probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.driven_motion import (
    DrivenModel, DrivenStudy, EventPolicy, FieldPhase, LinearDrag,
    UniformFieldSchedule, investigate_driven_motion,
)
from conductor_sim.driven_reassembly import reassemble_driven_model
from conductor_sim.driven_viewer import render_driven_run


def author():
    base = PrescribedScenario(
        "m26a:baseline",
        TestChargeMotionSpec(
            PrescribedField(()),
            PointCharge(1e-9, (0., 0., .1)),
            1e-6, (.01, 0., 0.), .04, .01,
            plane_normal=(0., 1., 0.),
            relative_tolerance=1e-9,
        ),
        (), "Teacher hypothetical: charge and initial motion are explicitly authored.",
    )
    parent = DrivenModel(
        base, LinearDrag(1e-8),
        UniformFieldSchedule((
            FieldPhase(0., (.1, 0., 0.)),
            FieldPhase(.02, (-.1, 0., .03)),
        )),
        "m26a:driven-original",
        "Teacher hypothetical: retained drag and scheduled field.",
    )
    assembly = reassemble_driven_model(
        parent,
        model_id="m26a:driven-point",
        scenario_id="m26a:base-point",
        note="Teacher hypothetical: added independent held point source.",
        add_sources=(NamedSource(
            "point:A", "Held point A", FixedPointField(1e-10, (.2, 0., .1))
        ),),
    )
    study = DrivenStudy(
        assembly.model, (), EventPolicy(.005, 1e-9, 1e-9, 1e-8),
    )
    return parent, assembly, investigate_driven_motion(study)


def export(output: Path):
    parent, assembly, result = author()
    assert parent.as_dict() == assembly.parent.as_dict()
    assert result.study.model is assembly.model
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_driven_run(result, title="M26a · Structural driven experiment"),
                      encoding="utf-8")
    metadata = dict(
        model=assembly.as_dict(),
        numerical_result=result.as_dict(),
        provenance="Teacher hypothetical; not a JEE source-given motion problem.",
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/m26a-driven.html"))
    args = parser.parse_args()
    metadata = export(args.output)
    print(json.dumps({
        "html": str(args.output),
        "child_revision": metadata["model"]["child_model_revision_id"],
        "termination": metadata["numerical_result"]["termination"],
        "switches": len(metadata["numerical_result"]["switches"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
