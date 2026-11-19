"""Independent stdlib-only M40a spherical induced sigma, identity and SI witness.

Never imports conductor_sim, the native spherical solver or the production
density formula. Local density is independently recovered as the negative
normal electric field of REAL and KELVIN IMAGE charges on the cavity wall.
Records prove mathematical consistency of teacher inputs, not provenance
of entirely self-consistently rewritten external JSON.
"""
from __future__ import annotations
import argparse
import json
from decimal import Decimal, localcontext
from math import fsum, hypot, isclose, isfinite, pi, ulp
from pathlib import Path

from tools.m39a_gauss_reference import check_gauss_record

SCHEMA="conductor_sim.m40a_spherical_multicavity_sigma.v2"
ROLE="teacher_hypothetical_spherical_multicavity_only"
DIRECTION_SEMANTICS=(
    "input_direction preserves original admitted near-unit binary64 components; "
    "local sigma follows their physical direction, normalized before wall evaluation; "
    "unit_normal is rounded display-only, not a precision source for near-wall sigma"
)
ASSUMPTIONS=[
    "one connected ideal metal with spherical outer surface and disjoint sealed spherical voids",
    "all point charges have verified positions strictly within their assigned spherical voids",
    "homogeneous positive permittivity, no exterior sources/fields, no dielectric interfaces",
    "only isolated fixed-metal-charge or fully grounded static equilibria",
    "spherical inner/outer surface sigma samples ONLY, not a 3D field or force solver",
]
UNSUPPORTED=[
    "irregular/non-spherical cavities, local 3D field/potential maps",
    "source forces, motion, ground-wire currents or transient grounding",
    "source-faithful solutions of workbook p11 or notes p70 irregular 3D local-density items",
]
UNITS={"charge":"C","position":"m","radius":"m","surface_density":"C/m^2"}


def required(obj,keys,label):
    if type(obj) is not dict or set(obj)!=set(keys):
        raise ValueError("changed geometry/SI evidence schema in "+label)


def real(x,label):
    if type(x) not in (int,float):
        raise ValueError(label+" must be a non-Boolean finite real")
    try:
        value=float(x)
    except (ValueError,OverflowError) as exc:
        raise ValueError(label+" is not representable as binary64") from exc
    if not isfinite(value):
        raise ValueError(label+" must be finite")
    return value


def coords(x,label):
    if type(x) is not list or len(x)!=3:
        raise ValueError(label+" requires exactly three saved coordinates")
    return tuple(real(v,label) for v in x)


def diff(a,b):
    return tuple(a[i]-b[i] for i in range(3))


def norm(a):
    return hypot(*a)


def density_from_images(R,sources,n,center):
    """Separate high-precision real+Kelvin-image normal E at the inner wall.

    High precision is important for nearly centered but nonzero source
    positions: a literal image may be 10**300 times farther away than the
    cavity radius, and forming that image in binary64 would overflow despite
    a perfectly finite, almost-uniform physical inner surface density.
    """
    with localcontext() as ctx:
        ctx.prec=90
        D=Decimal.from_float
        radius=D(R)
        direction=tuple(D(x) for x in n)
        direction_norm=(sum((x*x for x in direction),Decimal(0))).sqrt()
        direction=tuple(x/direction_norm for x in direction)
        wall=tuple(radius*v for v in direction)
        pi_d=D(pi)
        terms=[]
        for charge,absolute in sources:
            if charge==0.:
                continue
            q=D(charge)
            u=tuple(D(absolute[i])-D(center[i]) for i in range(3))
            d2=sum((v*v for v in u),Decimal(0))
            if d2==0:
                terms.append(-q/(4*pi_d*radius*radius))
                continue
            d=d2.sqrt()
            real_delta=tuple(wall[k]-u[k] for k in range(3))
            r2=sum((v*v for v in real_delta),Decimal(0))
            image=tuple(radius*radius*v/d2 for v in u)
            image_charge=-q*radius/d
            image_delta=tuple(wall[k]-image[k] for k in range(3))
            ri2=sum((v*v for v in image_delta),Decimal(0))
            real_dot=sum((direction[k]*real_delta[k] for k in range(3)),Decimal(0))
            image_dot=sum((direction[k]*image_delta[k] for k in range(3)),Decimal(0))
            # sigma_metal = -epsilon * (E_cavity dot n), where E includes
            # both REAL and IMAGE point-source electric fields.
            term=-(q*real_dot/(r2*r2.sqrt()) +
                   image_charge*image_dot/(ri2*ri2.sqrt()))/(4*pi_d)
            terms.append(term)
        actual=sum(terms,Decimal(0))
    try:
        as_float=float(actual)
    except (OverflowError,ValueError) as exc:
        raise ValueError("independent Kelvin image field overflows binary64") from exc
    if not isfinite(as_float) or (actual!=0 and as_float==0.):
        raise ValueError("independent Kelvin image field overflows or underflows binary64")
    return as_float


def same(actual,expected,label):
    got=real(actual,label)
    if not isfinite(expected) or (expected==0.)!=(got==0.):
        raise ValueError(label+" lost physical signed zero, or overflowed")
    if expected and ((expected>0)!=(got>0)):
        raise ValueError(label+" reversed signed surface charge")
    if expected:
        tolerance=min(64*ulp(abs(expected)),abs(expected)/4)
        if not isclose(got,expected,rel_tol=2e-10,abs_tol=tolerance):
            raise ValueError(label+" disagrees with independent Kelvin image field")


def check_record(doc):
    required(doc,(
        "schema","source_role","direction_interpretation","original_workbook_items_newly_solved",
        "assumptions","unsupported","geometry_qualified_spherical_only",
        "irregular_3d_fields_qualified","time_evolution_defined","units",
        "outer_sphere","voids","gauss","samples",
    ),"spherical sigma")
    if (doc["schema"]!=SCHEMA or doc["source_role"]!=ROLE
        or type(doc["original_workbook_items_newly_solved"]) is not int
        or doc["original_workbook_items_newly_solved"]!=0):
        raise ValueError("fake source role/solved JEE claim")
    if doc["direction_interpretation"]!=DIRECTION_SEMANTICS:
        raise ValueError("physical input direction provenance is falsified")
    if doc["assumptions"]!=ASSUMPTIONS or doc["unsupported"]!=UNSUPPORTED:
        raise ValueError("M40a spherical/unsupported physics boundary altered")
    if (doc["geometry_qualified_spherical_only"] is not True
        or doc["irregular_3d_fields_qualified"] is not False
        or doc["time_evolution_defined"] is not False
        or type(doc["units"]) is not dict or doc["units"]!=UNITS):
        raise ValueError("fake geometry/surface units/time qualification")
    gauss=doc["gauss"]
    check_gauss_record(gauss)
    metal=doc["outer_sphere"]
    required(metal,("center_m","radius_m"),"outer metal")
    center=coords(metal["center_m"],"outer center")
    outer=real(metal["radius_m"],"outer radius")
    if not 1e-6<=outer<=1e6 or any(abs(x)>outer*1e6 for x in center):
        raise ValueError("outer spherical geometry is not in qualified scale")
    voids=doc["voids"]
    if type(voids) is not list or len(voids)!=len(gauss["cavities"]):
        raise ValueError("missing/extra sealed physical void")
    positions={}
    charge_spread=[]
    for i,(row,original) in enumerate(zip(voids,gauss["cavities"])):
        required(row,("cavity_id","center_m","radius_m","source_positions"),"inner void")
        if row["cavity_id"]!=original["cavity_id"]:
            raise ValueError("false cavity physical identity/order")
        if row["cavity_id"]=="outer":
            raise ValueError("reserved outer surface identity cannot be an inner cavity")
        c=coords(row["center_m"],"cavity center")
        r=real(row["radius_m"],"cavity radius")
        if not 1e-6<=r<=1e6 or any(abs(x)>outer*1e6 for x in c):
            raise ValueError("unqualified cavity radius/world coordinates")
        if norm(diff(c,center))+r>=outer*(1-1e-8):
            raise ValueError("spherical cavity touches/escapes outer metal")
        listed=row["source_positions"]
        sources=original["authored_sources"]
        if type(listed) is not list or len(listed)!=len(sources):
            raise ValueError("source geometry is not bijective with Gauss inventory")
        actual={}
        for entry in listed:
            required(entry,("source_id","position_m"),"named source geometry")
            ident=entry["source_id"]
            if type(ident) is not str or ident in actual:
                raise ValueError("invalid or duplicated source placement identity")
            p=coords(entry["position_m"],"source position")
            if any(abs(x)>outer*1e6 for x in p) or norm(diff(p,c))>=r*(1-1e-8):
                raise ValueError("point source lies outside/on/near spherical cavity wall")
            actual[ident]=p
        if set(actual)!={s["source_id"] for s in sources}:
            raise ValueError("source identity placement/charge mismatch")
        named=[]
        for s in sources:
            q=real(s["charge_c"],"source Coulombs")
            named.append((q,actual[s["source_id"]]))
            if q:charge_spread.append(abs(q))
        positions[row["cavity_id"]]=(r,c,named)
        for before in voids[:i]:
            cc=coords(before["center_m"],"previous cavity center")
            rr=real(before["radius_m"],"previous cavity radius")
            if norm(diff(c,cc))<=r+rr+outer*1e-8:
                raise ValueError("physical spherical cavity overlap or contact")
    if charge_spread and (min(charge_spread)<1e-240
        or max(charge_spread)>1e180
        or max(charge_spread)/min(charge_spread)>1e20):
        raise ValueError("signed local sigma charge spread outside qualified range")
    samples=doc["samples"]
    if type(samples) is not list or len(samples)>64:
        raise ValueError("unknown or too many local surface inquiries")
    for row in samples:
        required(row,("surface_id","input_direction","unit_normal","sigma_c_per_m2"),
                 "local density probe")
        raw=coords(row["input_direction"],"encoded physical input direction")
        norm_raw=norm(raw)
        if not isfinite(norm_raw) or abs(norm_raw-1)>1e-12:
            raise ValueError("original encoded direction is not near unit")
        display=coords(row["unit_normal"],"rounded display-only unit normal")
        # Compute from the ORIGINAL binary64-encoded input, using Decimal
        # only for the direction-to-unit operation; do not use the displayed
        # binary64-normalized vector to certify near-wall physics.
        with localcontext() as ctx:
            ctx.prec=90
            D=Decimal.from_float
            v=tuple(D(x) for x in raw)
            L=(sum((x*x for x in v),Decimal(0))).sqrt()
            expected_display=tuple(float(x/L) for x in v)
        if display!=expected_display:
            raise ValueError("unit-normal display is not consistent with original input direction")
        surface=row["surface_id"]
        if surface=="outer" and type(surface) is str:
            Q=real(gauss["outer_face_total_charge_c"],"outer Gauss charge")
            expected=Q/(4*pi*outer*outer)
        elif type(surface) is str and surface in positions:
            R,center,sources=positions[surface]
            expected=density_from_images(R,sources,raw,center)
        else:
            raise ValueError("unknown spherical physical surface ID")
        same(row["sigma_c_per_m2"],expected,"independent surface density")
    return {"schema":"conductor_sim.m40a_independent_sigma_witness.v2",
            "teacher_hypothetical_assemblies":1,
            "geometry_certified_spherical_only":True,
            "cavities_checked":len(voids),
            "point_sources_checked":len({s["source_id"] for x in gauss["cavities"] for s in x["authored_sources"]}),
            "sampled_local_surface_densities_checked":len(samples),
            "new_original_jee_solved":0,"arbitrary_3d_fields_certified":False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    x=json.loads(args.input.read_text(encoding="utf-8"))
    result=check_record(x)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
