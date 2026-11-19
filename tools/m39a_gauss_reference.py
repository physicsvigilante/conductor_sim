"""Independent stdlib-only Gauss-law theorem and typed-evidence checker (M39a).

NO conductor_sim imports; uses Decimal for algebraic sums independently from
production's math.fsum, and strictly distinguishes bool from zero. This checks
a declared source/metal record's mathematical consistency, not proof that a
teacher's topology declaration corresponds to a verified geometric cavity.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, localcontext
import json
from math import isclose, isfinite, ulp
from pathlib import Path
import re


SCHEMA = "conductor_sim.gauss_cavity_network.v1"
ROLE = "teacher_hypothetical_topology_gauss_only"
ANALYSIS = "conditional exact Gauss-law conductor-face charge TOTALS, never local geometry"
MEMBERSHIP = (
    "TEACHER-DECLARED membership in mutually sealed noncontacting cavities; "
    "no geometry or containment was verified"
)
ASSUMPTIONS = [
    "one bounded connected ideal electrostatic metal with 1..8 disjoint fully sealed voids",
    "all charges are fixed inside their declared void and do not contact metal",
    "no exterior sources, applied exterior field, other metal, internal electrode or aperture",
    "homogeneous positive permittivity; vacuum/infinity reference V=0 for grounding",
    "no time evolution; each revised assembly is a separate static hypothetical",
]
UNSUPPORTED = [
    "pointwise electric field E(x), including within individual cavities",
    "pointwise potential V(x), except the grounded metal boundary theorem",
    "surface charge density sigma(x) or arbitrary-shaped induced charge map",
    "geometric cavity containment, radius, area, mesh and visualization",
    "source force, energy, wire current, discharge time, heat or radiation",
]
UNITS = {"charge": "C", "permittivity": "F/m", "exterior_electric_flux": "V*m"}
ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")


def _required(record, keys, what):
    if type(record) is not dict or set(record) != set(keys):
        raise ValueError(f"{what} has an unexpected SI/provenance/identity schema")


def _real(value, what):
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError(f"{what} must be a finite non-Boolean SI real")
    return Decimal(str(value))


def _id(value, what):
    if type(value) is not str or ID.fullmatch(value) is None:
        raise ValueError(f"{what} does not have a stable physical identity")


def _same(value, expected: Decimal, what):
    got=_real(value, what)
    expected_float=float(expected)
    if not isfinite(expected_float):
        raise ValueError(f"{what} exceeded permitted float/physics domain")
    if expected_float == 0:
        if float(got) != 0:
            raise ValueError(f"{what} differs from exact zero Gauss-law integral")
        return
    tolerance=64*ulp(abs(expected_float))
    if not isclose(float(got),expected_float,rel_tol=8e-13,abs_tol=tolerance):
        raise ValueError(f"{what} differs from independently derived Gauss-law SI total")


def check_gauss_record(doc):
    _required(doc, (
        "schema","analysis","source_role","original_workbook_items_newly_solved",
        "membership_provenance","assumptions","unsupported","units",
        "conductor","permittivity_f_per_m","cavities",
        "total_interior_source_charge_c","outer_face_total_charge_c",
        "exterior_electric_flux_v_m","exterior_electric_field_zero_theorem",
        "geometry_verified","source_position_defined","time_evolution_defined",
    ), "Gauss-only result")
    if doc["schema"]!=SCHEMA or doc["analysis"]!=ANALYSIS:
        raise ValueError("Gauss-only schema or exact conditional theorem changed")
    if (doc["source_role"]!=ROLE or doc["membership_provenance"]!=MEMBERSHIP
            or type(doc["original_workbook_items_newly_solved"]) is not int
            or doc["original_workbook_items_newly_solved"]!=0):
        raise ValueError("fabricated original workbook/source/topology provenance")
    if doc["assumptions"]!=ASSUMPTIONS or doc["unsupported"]!=UNSUPPORTED:
        raise ValueError("Gauss-only topology qualification or refused physics was altered")
    if type(doc["units"]) is not dict or doc["units"]!=UNITS:
        raise ValueError("wrong physical SI units")
    for k in ("geometry_verified","source_position_defined","time_evolution_defined"):
        if doc[k] is not False:
            raise ValueError("claimed geometry, position or temporal physics is unsupported")

    eps=_real(doc["permittivity_f_per_m"],"homogeneous permittivity")
    if eps<=0:
        raise ValueError("permittivity must be strictly positive")
    metal=doc["conductor"]
    _required(metal,(
        "conductor_id","constraint","authored_isolated_metal_charge_c",
        "equilibrium_net_metal_charge_c",
    ),"single connected metal")
    _id(metal["conductor_id"],"connected conductor")
    mode=metal["constraint"]
    if mode not in ("isolated_fixed_net_charge","grounded_zero_potential") or type(mode) is not str:
        raise ValueError("unsupported connected-metal electrical constraint")
    grounded=mode=="grounded_zero_potential"
    authored=metal["authored_isolated_metal_charge_c"]
    if grounded:
        if authored is not None:
            raise ValueError("grounded metal cannot also assert an isolated fixed charge")
    else:
        _real(authored,"isolated metal fixed net charge")

    cavities=doc["cavities"]
    if type(cavities) is not list or not 1<=len(cavities)<=8:
        raise ValueError("only 1..8 declared sealed cavities qualified")
    ids={metal["conductor_id"]}
    totals=[]
    number=0
    for idx,row in enumerate(cavities):
        _required(row,(
            "cavity_id","sealed","authored_sources",
            "source_charge_sum_c","induced_inner_face_total_charge_c",
        ),"cavity integral")
        _id(row["cavity_id"],f"cavity {idx}")
        if row["cavity_id"] in ids:
            raise ValueError("duplicate cavity or metal physical identity")
        ids.add(row["cavity_id"])
        if row["sealed"] is not True:
            raise ValueError("Gauss integral requires genuine authored sealed cavity")
        sources=row["authored_sources"]
        if type(sources) is not list or len(sources)>16:
            raise ValueError("cavity source identity list malformed")
        source_totals=[]
        for j,source in enumerate(sources):
            _required(source,("source_id","charge_c"),"source")
            _id(source["source_id"],f"cavity {idx} source {j}")
            if source["source_id"] in ids:
                raise ValueError("multiply assigned physical source identity")
            ids.add(source["source_id"])
            source_totals.append(_real(source["charge_c"],"actual enclosed source charge"))
            number+=1
        q=sum(source_totals,Decimal(0))
        _same(row["source_charge_sum_c"],q,"independent cavity enclosed charge")
        _same(row["induced_inner_face_total_charge_c"],-q,
              "induced inner-face net charge")
        totals.append(q)
    if number>16:
        raise ValueError("too many source identities in one metal")
    total=sum(totals,Decimal(0))
    _same(doc["total_interior_source_charge_c"],total,"total enclosed cavity sources")
    metal_expected=-total if grounded else _real(authored,"fixed metal charge")
    outer_expected=Decimal(0) if grounded else metal_expected+total
    _same(metal["equilibrium_net_metal_charge_c"],metal_expected,
          "actual metal net equilibrium charge")
    _same(doc["outer_face_total_charge_c"],outer_expected,"entire outer metal face")
    if doc["exterior_electric_field_zero_theorem"] is not grounded:
        raise ValueError("invalid grounded no-exterior-source uniqueness theorem claim")
    with localcontext() as ctx:
        ctx.prec=70
        try:
            flux=outer_expected/eps
        except (InvalidOperation,ZeroDivisionError,OverflowError) as exc:
            raise ValueError("not representable exterior electric flux") from exc
    _same(doc["exterior_electric_flux_v_m"],flux,
          "Gauss exterior electric flux with correct V*m SI")
    return {
        "schema":"conductor_sim.m39a_independent_gauss_oracle.v1",
        "teacher_hypothetical_assemblies":1,
        "original_workbook_items_newly_solved":0,
        "cavities_independently_checked":len(cavities),
        "point_source_identities_checked":number,
        "metal_constraint":mode,
        "total_source_charge_c":float(total),
        "induced_outer_total_charge_c":float(outer_expected),
        "gauss_flux_v_m":float(flux),
        "local_3d_fields_qualified":False,
        "physical_containment_geometrically_verified":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    actual=json.loads(a.input.read_text(encoding="utf-8"))
    report=check_gauss_record(actual)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
