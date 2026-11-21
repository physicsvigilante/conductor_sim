"""M40c independent witness consistency, using prior real + Kelvin image oracle.

No production M40c or conductor_sim physics is imported. A wholly rewritten
self-consistent third-party record is NOT authenticated to teacher history.
"""
from __future__ import annotations
import argparse
import json
from math import isclose, isfinite, ulp
from pathlib import Path

from tools.m40a_spherical_reference import check_record, density_from_images, required

SCHEMA = "conductor_sim.m40c_static_spherical_relocation.v1"
ROLE = "teacher_hypothetical_static_spherical_source_relocation_only"
NO_CHRONOLOGY = (
    "Before and after are separately qualified STATIC hypothetical equilibria; "
    "they do not define a physical motion path, elapsed time, grounding current, "
    "force, work or energy transfer."
)


def _e(x, y, label):
    def canon(v):
        return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False)
    if canon(x)!=canon(y):
        raise ValueError(label+" contradicts actual qualified static source evidence")


def check_static_comparison(doc):
    if type(doc) is not dict or doc.get("schema") != SCHEMA or doc.get("source_role") != ROLE:
        raise ValueError("not genuine versioned M40c teacher-only comparison")
    for flag,value in (("original_workbook_items_newly_solved",0),
                       ("time_evolution_defined",False),
                       ("geometry_qualified_spherical_only",True),
                       ("local_3d_fields_or_forces_qualified",False)):
        _e(doc.get(flag),value,"source or physics claim "+flag)
    _e(doc.get("units"),{"position":"m","charge":"C","surface_density":"C/m^2"},"SI units")
    _e(doc.get("no_chronology"),NO_CHRONOLOGY,"no chronology honesty")
    if set(doc) != {"schema", "source_role", "original_workbook_items_newly_solved",
                    "time_evolution_defined", "no_chronology",
                    "geometry_qualified_spherical_only", "local_3d_fields_or_forces_qualified",
                    "units", "relocation", "outer_sphere", "baseline_voids", "revised_voids",
                    "gauss", "cavities", "outer_uniform_sigma_c_per_m2",
                    "m40a_v2_witness_records"}:
        raise ValueError("no invented additional physical capabilities or witness fields")
    witness=doc.get("m40a_v2_witness_records")
    if type(witness) is not dict or set(witness)!={"baseline","revised"}:
        raise ValueError("two authentic independently checkable static states required")
    base,rev=witness["baseline"],witness["revised"]
    if (type(base) is not list or type(rev) is not list or not 2 <= len(base) <= 9
            or len(base)!=len(rev)):
        raise ValueError("all cavity faces and outer witness records required")
    for record in base+rev:
        check_record(record) # real+image independent boundary electric field
    for name,records in (("baseline",base),("revised",rev)):
        for x in records[1:]:
            _e(x["gauss"],records[0]["gauss"],name+" Gauss inventory")
            _e(x["voids"],records[0]["voids"],name+" 3D void geometry")
            _e(x["outer_sphere"],records[0]["outer_sphere"],name+" 3D outer geometry")
    _e(base[0]["gauss"],rev[0]["gauss"],"unchanged charge/Gauss identity")
    _e(base[0]["outer_sphere"],rev[0]["outer_sphere"],"unchanged outer sphere")
    _e(doc.get("gauss"),base[0]["gauss"],"visible original Gauss ledger")
    _e(doc.get("outer_sphere"),base[0]["outer_sphere"],"visible outer geometry")
    _e(doc.get("baseline_voids"),base[0]["voids"],"visible original 3D voids")
    _e(doc.get("revised_voids"),rev[0]["voids"],"visible modified 3D voids")
    voids=base[0]["voids"]
    if (type(doc.get("cavities")) is not list or
            len(doc["cavities"]) != len(voids) or len(base)!=len(voids)+1):
        raise ValueError("all original cavity rows required")
    relocated=doc.get("relocation")
    if type(relocated) is not dict or set(relocated)!={"source_id","cavity_id","original_position_m","revised_position_m"}:
        raise ValueError("exact source identity/position revision required")
    changes=[]
    for a,b in zip(base[0]["voids"],rev[0]["voids"]):
        _e([a["cavity_id"],a["center_m"],a["radius_m"]],
           [b["cavity_id"],b["center_m"],b["radius_m"]],"unchanged spherical boundaries")
        if len(a["source_positions"])!=len(b["source_positions"]):
            raise ValueError("source count may not change")
        for sa,sb in zip(a["source_positions"],b["source_positions"]):
            _e(sa["source_id"],sb["source_id"],"stable original named source")
            if sa["position_m"]!=sb["position_m"]:
                changes.append({"source_id":sa["source_id"],"cavity_id":a["cavity_id"],
                                "original_position_m":sa["position_m"],
                                "revised_position_m":sb["position_m"]})
    if len(changes)!=1:
        raise ValueError("exactly one physical source must be relocated")
    _e(relocated,changes[0],"visible source relocation")
    for i,(cav,a,b) in enumerate(zip(doc["cavities"],base,rev)):
        required(cav, ("center_m","radius_m","cavity_id","whole_face_induced_charge_c","samples"), "comparison cavity")
        if type(cav) is not dict or a["voids"][i]["cavity_id"]!=cav.get("cavity_id"):
            raise ValueError("cavity order/ID mismatches original physical geometry")
        if any(x["surface_id"]!=cav["cavity_id"] for x in a["samples"]+b["samples"]):
            raise ValueError("probe surface identity mismatch")
        _e(cav.get("center_m"),voids[i]["center_m"],"visible center")
        _e(cav.get("radius_m"),voids[i]["radius_m"],"visible radius")
        _e(cav.get("whole_face_induced_charge_c"),
           doc["gauss"]["cavities"][i]["induced_inner_face_total_charge_c"],
           "whole induced Gauss face")
        ss=cav.get("samples")
        if type(ss) is not list or not 1 <= len(ss) <= 24 or len(ss)!=len(a["samples"]) or len(ss)!=len(b["samples"]):
            raise ValueError("all admitted point rays required")
        for item,aa,bb in zip(ss,a["samples"],b["samples"]):
            required(item, ("input_direction","unit_normal_display_only","sigma_baseline_c_per_m2",
                            "sigma_revised_c_per_m2","delta_status","delta_c_per_m2"), "comparison sample")
            _e(aa["input_direction"],bb["input_direction"],"same original physical ray")
            _e(item.get("input_direction"),aa["input_direction"],"raw physical ray")
            _e(item.get("unit_normal_display_only"),aa["unit_normal"],"display-only rounded normal")
            _e(item.get("sigma_baseline_c_per_m2"),aa["sigma_c_per_m2"],"original image sigma")
            _e(item.get("sigma_revised_c_per_m2"),bb["sigma_c_per_m2"],"revised image sigma")
            x,y=aa["sigma_c_per_m2"],bb["sigma_c_per_m2"]
            if cav["cavity_id"] != relocated["cavity_id"]:
                if x!=y:raise ValueError("other cavity local wall density changed")
                expected_status,delta="unchanged_exact",0.0
            else:
                d=y-x
                if not isfinite(d):raise ValueError("overflowed signed difference")
                q = next(s["charge_c"] for s in doc["gauss"]["cavities"][i]["authored_sources"]
                         if s["source_id"] == relocated["source_id"])
                old, new = [density_from_images(cav["radius_m"], [(q, relocated[key])],
                                                aa["input_direction"], cav["center_m"])
                            for key in ("original_position_m", "revised_position_m")]
                independent_change = new - old
                if (abs(independent_change) <= 32 * ulp(max(abs(x), abs(y), abs(old), abs(new)))
                        or not isclose(d, independent_change, rel_tol=2e-10, abs_tol=0.0)):
                    expected_status,delta="unresolved_numerical_difference",None
                else:expected_status,delta="resolved_difference",d
            _e(item.get("delta_status"),expected_status,"numerical difference status")
            _e(item.get("delta_c_per_m2"),delta,"signed difference witness")
    for x in (base[-1],rev[-1]):
        if len(x["samples"])!=1 or x["samples"][0]["surface_id"]!="outer":
            raise ValueError("true outer uniform spherical density record required")
    _e(base[-1]["samples"],rev[-1]["samples"],"unchanged outer sigma")
    _e(doc.get("outer_uniform_sigma_c_per_m2"),base[-1]["samples"][0]["sigma_c_per_m2"],"outer sigma")
    return {"status":"PASS","cavities_checked":len(voids),"inner_samples_checked":sum(len(x["samples"]) for x in base[:-1])*2,
            "one_source_relocated":True,"original_jee_solutions":0,
            "external_html_cryptographically_authenticated":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    data=check_static_comparison(json.loads(args.input.read_text(encoding="utf-8")))
    if args.output:args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(data,sort_keys=True))


if __name__=="__main__":main()
