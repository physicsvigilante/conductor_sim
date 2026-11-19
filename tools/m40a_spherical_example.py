"""Teacher-authored actual M40a spherical multicavity surface-density inquiry example."""
from __future__ import annotations

from dataclasses import replace
import argparse
import json
from math import isclose
from pathlib import Path

from conductor_sim.cavity_gauss import (
    CavityCharge, SealedCavity, GaussCavityAssembly,
)
from conductor_sim.model import FixedNetCharge, Grounded, EPSILON_0
from conductor_sim.spherical_multicavity import (
    NamedSourcePosition, SphericalVoid, SphericalMetalGeometry,
    SphericalSigmaModel, solve_spherical_sigma,
)


def teacher_model() -> SphericalSigmaModel:
    gauss = GaussCavityAssembly(
        "MetalSphere", (
            SealedCavity("NeutralButNonuniform", (
                CavityCharge("plus_A", +2e-9),
                CavityCharge("minus_A", -2e-9),
            )),
            SealedCavity("NegativeCavity", (
                CavityCharge("minus_B", -3e-9),
                CavityCharge("plus_B", +1e-9),
            )),
            SealedCavity("EmptyCavity"),
        ), FixedNetCharge(+1e-9), EPSILON_0,
    )
    geom = SphericalMetalGeometry(
        (0., 0., 0.), 5., (
            SphericalVoid("NeutralButNonuniform", (-1.7, 0., 0.), .65, (
                NamedSourcePosition("plus_A", (-1.4, 0., 0.)),
                NamedSourcePosition("minus_A", (-1.95, 0., 0.)),
            )),
            SphericalVoid("NegativeCavity", (1.5, .8, 0.), .8, (
                NamedSourcePosition("minus_B", (1.6, .95, .05)),
                NamedSourcePosition("plus_B", (1.2, .6, -.2)),
            )),
            SphericalVoid("EmptyCavity", (0., -1.8, 0.), .5),
        ),
    )
    return SphericalSigmaModel(gauss, geom)


def run_all(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    model = teacher_model()
    normals = ((1.,0.,0.),(-1.,0.,0.),(0.,1.,0.))
    probes = [
        ("NeutralButNonuniform",normals[0]),
        ("NeutralButNonuniform",normals[1]),
        ("NegativeCavity",normals[0]),
        ("NegativeCavity",normals[2]),
        ("EmptyCavity",normals[0]),
        ("outer",normals[0]),("outer",normals[1]),
    ]
    result = {}
    for name, constraint in (
        ("isolated", model.gauss_assembly.constraint),
        ("grounded", Grounded()),
    ):
        spec = replace(model, gauss_assembly=replace(model.gauss_assembly,
                                                    constraint=constraint))
        sol = solve_spherical_sigma(spec)
        data = sol.as_dict(probes)
        (directory / ("m40a-" + name + "-teacher.json")).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
            encoding="utf-8")
        result[name] = data
    assert isclose(result["isolated"]["gauss"]["outer_face_total_charge_c"], -1e-9, rel_tol=1e-14)
    assert result["grounded"]["gauss"]["outer_face_total_charge_c"] == 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m40a"))
    args=parser.parse_args()
    result=run_all(args.output_dir)
    print("Teacher-only spherical sigma samples:", args.output_dir)
    print("Cavities:",len(result["isolated"]["voids"]),"new original JEE answers: 0")


if __name__=="__main__":
    main()
