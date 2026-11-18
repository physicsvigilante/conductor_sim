"""M36d: inspect REAL source-authentic SOLID conductor HTML in offline Chromium.

The numerical Gauss-law reference belongs to an entirely separate process.
This script verifies actual source/teacher static-state visibility and genuine
SOLID disk geometry on the viewer's *embedded* live frame data, no network.
"""
from __future__ import annotations

import argparse
import json
import re
from math import isclose
from pathlib import Path

from tools.m36e_embedded_grid_reference import check_embedded_solid_grid
from tools.m36d_solid_reference import check as independently_check_native_sphere

# Strictly scoped to printed p26, not other M5 viewers. This fixed wording
# separates the source's SYMBOLIC givens from all teacher-chosen numeric data.
SOURCE_INTRO = (
    "The printed source supplies SYMBOLIC R and Q. "
    "All numerical inputs are teacher-chosen. "
    "Switching the stored-state slider updates these values to the "
    "selected genuine static solution, NOT a time trajectory."
)
SOURCE_BANNER = (
    "SOURCE GEOMETRY: genuinely SOLID conducting sphere. "
    "The original p26 Illustration 1 asks V(R/2) and V(3R/2) "
    "of a charged SOLID conductor, and zero E inside its metal. "
    "The first stored state uses REAL SolidSphere physics and a solid "
    "disk cross-section, NOT the historical hollow-shell proxy. "
    "A TRUE SOLID has ONE outer charged face and NO physical inner face; "
    "an inner charge of zero is only a result bookkeeping value. "
    "Printed values R and Q are SYMBOLIC. The R=0.10m, Q=+1nC "
    "numbers are TEACHER illustration choices. "
    "Further stored states are labelled TEACHER HYPOTHETICAL: "
    "a changed signed-charge/radius/centre world and a grounded "
    "source-free world. The slider changes PRECOMPUTED STATIC states, "
    "NOT elapsed-time charge transfer."
)
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
CAPTION = re.compile(
    r"TRUE SOLID Sphere A: R=(" + NUM + r") m; (.*?)"
    r"; center=\((" + NUM + r"), (" + NUM + r"), (" + NUM + r")\) m\. "
    r"Only one physical OUTER charge-density face exists\."
)


def _check_source_caption(actual_text: str, state: dict, index: int) -> None:
    """Validate the VISIBLE charge, R and centre, not only metadata/keywords."""
    m=CAPTION.fullmatch(actual_text.strip())
    if m is None:
        raise AssertionError("visible source sphere radius/charge/centre caption malformed or falsified")
    radius,charge_text,cx,cy,cz=m.groups()
    geom=state["physical_geometry"]
    if not isclose(float(radius),geom["radius_m"],rel_tol=3e-10,abs_tol=1e-14):
        raise AssertionError("visible source solid radius contradicts native geometry")
    for j,(visible,expected) in enumerate(zip((cx,cy,cz),geom["center_m"])):
        if not isclose(float(visible),expected,rel_tol=3e-10,abs_tol=1e-14):
            raise AssertionError("visible sphere world centre contradicts actual physical geometry axis "+str(j))
    result=state["observed_conductor_results"][0]
    if state["constraint_type"]=="Grounded":
        if charge_text!="Grounded · actual solved Q=0 C in empty space":
            raise AssertionError("visible grounded sphere charge caption contradicts source-free Q=0")
        if state["authored_charge_c"] is not None or not isclose(
            result["net_charge_c"],0.,rel_tol=0.,abs_tol=1e-21
        ):
            raise AssertionError("grounded source manifest falsely claims nonzero net charge")
    else:
        cm=re.fullmatch(r"Isolated net Q=(" + NUM + r") nC \(authored teacher value\)",charge_text)
        if cm is None:
            raise AssertionError("visible isolated net-charge caption malformed or given false charge")
        actual_q=float(cm.group(1))*1e-9
        for expected in (result["net_charge_c"],state["authored_charge_c"]):
            if not isclose(actual_q,expected,rel_tol=3e-10,abs_tol=1e-20):
                raise AssertionError("visible isolated signed Q contradicts actual native physical charge")



def check(directory:Path)->dict:
    from playwright.sync_api import sync_playwright

    manifest=json.loads((directory/"native-solid-sphere.json").read_text(encoding="utf-8"))
    if manifest["schema"]!="conductor_sim.m36d_real_solid_p26.v1":
        raise AssertionError("not actual source-authentic solid p26 output")
    filename=directory/manifest["viewer_file"]
    if not filename.is_file():
        raise AssertionError("source SOLID viewer HTML not actually written")
    screenshot_dir=directory/"screenshots"
    screenshot_dir.mkdir(exist_ok=True)
    recorded=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        try:
            page=browser.new_page(viewport={"width":1280,"height":820},device_scale_factor=1)
            external_requests=[]
            js_errors=[]
            page.on("request",lambda r: external_requests.append(r.url)
                    if r.url.startswith(("http://","https://")) else None)
            page.on("pageerror",lambda e: js_errors.append(str(e)))
            try:
                page.goto(filename.resolve().as_uri(),wait_until="load",timeout=45000)
                if page.evaluate("window.conductorViewer?.frameCount")!=3:
                    raise AssertionError("real static original/teacher frames missing")
                if page.evaluate("window.conductorViewer?.mode")!="static":
                    raise AssertionError("no time motion in original static solid p26")
                banner=page.locator("#m36d-source-authentic-solid")
                if not banner.is_visible():
                    raise AssertionError("printed source authenticity note not visually present")
                banner_text=banner.inner_text()
                if banner_text.strip()!=SOURCE_BANNER:
                    raise AssertionError(
                        "source-authentic banner falsely changes original symbolic/teacher provenance"
                    )
                if page.locator("canvas#scene").count()!=1:
                    raise AssertionError("no genuine painted physical conductor canvas")
                answer_panel=page.locator("#m36e-source-observations")
                if not answer_panel.is_visible():
                    raise AssertionError("JEE original source potential readouts not visibly presented")
                if "Original JEE p26: native potential observations" not in answer_panel.inner_text():
                    raise AssertionError("original textbook question not visibly identified")
                if answer_panel.locator("p").first.inner_text().strip()!=SOURCE_INTRO:
                    raise AssertionError(
                        "source-given symbols and teacher numbers not distinguished or falsely specified"
                    )
                if page.locator("#density-panel h2").inner_text()!="Surface-charge density (one real outer face)":
                    raise AssertionError("isolated fixed-Q sigma mislabeled induced charge")
                if page.locator("#profile").get_attribute("aria-label")!="Surface-charge density along a great circle":
                    raise AssertionError("density chart accessible label is physically misleading")
                # The embedded JSON is the data ACTUALLY consumed by the browser.
                # Independent SI checks cover every one of its 65x65 potential
                # and signed 3D field values, not just array dimensions.
                embedded=page.evaluate(
                    "() => JSON.parse(document.getElementById('simulation-data').textContent)"
                )
                grid_reference=check_embedded_solid_grid(manifest,embedded)
                cards=answer_panel.locator("[data-m36e-state]")
                if cards.count()!=3:
                    raise AssertionError("source answer card missing one of its three true physics states")
                for i,state in enumerate(manifest["native_states"]):
                    page.locator("#frame").evaluate(
                        "(el,index) => {el.value=String(index);"
                        "el.dispatchEvent(new Event('input',{bubbles:true}));}",i
                    )
                    seen=page.locator("#case").inner_text(timeout=15000)
                    if seen!=state["state_title"]:
                        raise AssertionError(f"visible physical state #{i} incorrectly labelled: {seen!r}")
                    if i==0 and "PRINTED p26 SOLID sphere" not in seen:
                        raise AssertionError("source ORIGINAL state not recognisable")
                    if i>0 and "TEACHER HYPOTHETICAL" not in seen:
                        raise AssertionError("changed physical world mistaken for source givens")
                    if page.locator("#density option").count()!=1:
                        raise AssertionError("solid surface must have ONE outer density profile")
                    active=cards.nth(i)
                    for candidate_index in range(3):
                        visible=cards.nth(candidate_index).is_visible()
                        if visible!=(candidate_index==i):
                            raise AssertionError("visible source answer card is stale or mismatched to slider")
                    if active.get_attribute("data-m36e-role")!=state["case_role"]:
                        raise AssertionError("answer card mistaken original textbook for teacher hypothesis")
                    answer_text=active.inner_text()
                    if i==0:
                        if "PRINTED ORIGINAL GEOMETRY" not in answer_text:
                            raise AssertionError("original formula answer card not identified")
                    elif "TEACHER HYPOTHETICAL" not in answer_text:
                        raise AssertionError("teacher changes falsely displayed as original source givens")
                    if i==2 and "Grounded" not in answer_text:
                        raise AssertionError("source-free grounded world was misattributed as fixed-Q")
                    if i!=2 and "Isolated net Q=" not in answer_text:
                        raise AssertionError("charge-controlled physical world lost isolation status")
                    paragraphs=active.locator("p")
                    if paragraphs.count()!=2:
                        raise AssertionError("source charge and geometry caption missing or duplicated")
                    _check_source_caption(paragraphs.nth(1).inner_text(),state,i)
                    native=state["native_samples"]
                    required={
                        "V_half_R":(native["half_radius_potential_v"],"V"),
                        "V_three_half_R":(native["three_half_radius_potential_v"],"V"),
                        "E_inside":(
                            sum(v*v for v in native["metal_half_radius_e_v_m"])**.5,
                            "V/m"),
                        "sigma_outer":(native["outer_sigma_c_m2"],"C/m²"),
                    }
                    if active.locator("[data-m36e-number]").count()!=len(required):
                        raise AssertionError("two original V observables or other actual SI values missing")
                    for key,(expected,unit) in required.items():
                        label=active.locator('[data-m36e-number="'+key+'"]')
                        if label.count()!=1:
                            raise AssertionError("missing/duplicated source native observation "+key)
                        raw=label.get_attribute("data-native-value")
                        if raw is None or not isclose(float(raw),expected,rel_tol=2e-12,abs_tol=1e-19):
                            raise AssertionError("display-native source evidence mismatch "+key)
                        text_value=label.inner_text()
                        if not text_value.endswith(" "+unit):
                            raise AssertionError("wrong actual physical units on source answer "+key)
                        if not isclose(float(text_value.split()[0]),expected,rel_tol=2e-9,abs_tol=1e-13):
                            raise AssertionError("visible original source answer is stale/forged "+key)
                    actual_text=page.locator("#density option").first.inner_text()
                    if "Sphere A / outer" not in actual_text:
                        raise AssertionError("imaginary inner surface in actual solid density viewer")
                    frame=page.evaluate(
                        "(index)=>{const bundle=JSON.parse("
                        "document.getElementById('simulation-data').textContent);"
                        "const f=bundle.frames[index];return {"
                        "geom:f.geometry,profiles:f.profiles,"
                        "conductors:f.conductor_results,source_masks:f.source_masks,"
                        "regions:f.grid.region, gridSize:[f.grid.x_m.length,f.grid.y_m.length],"
                        "label:f.label};}",i
                    )
                    if len(frame["geom"])!=1:
                        raise AssertionError("source shape has phantom extra object")
                    shape=frame["geom"][0]
                    expected=manifest["native_states"][i]["physical_geometry"]
                    if shape["kind"]!="sphere" or shape["name"]!="Sphere A":
                        raise AssertionError("real canvas is not the named source sphere")
                    if shape["inner_radius_m"]!=0 or shape["outer_radius_m"]!=expected["radius_m"]:
                        raise AssertionError("viewer draws HOLLOW shell rather than true SOLID")
                    if len(frame["profiles"])!=1 or frame["profiles"][0]["surface"]!="outer":
                        raise AssertionError("nonphysical inner surface profile shown")
                    if len(frame["conductors"])!=1 or frame["conductors"][0]["name"]!="Sphere A":
                        raise AssertionError("wrong named physical conductor in visualization")
                    if frame["source_masks"]:
                        raise AssertionError("fake point source/singularity inside solid")
                    if not any("metal" in str(region).lower() for row in frame["regions"] for region in row):
                        raise AssertionError("real solid interior never painted as metal")
                    if frame["gridSize"]!=[65,65]:
                        raise AssertionError("new denser grid must be genuine 65x65 physical samples")
                    if page.locator("#scene").is_visible() is False:
                        raise AssertionError("real canvas hidden")
                    saved=screenshot_dir/f"state-{i}.png"
                    page.screenshot(path=str(saved),full_page=True,animations="disabled")
                    recorded.append({
                        "state":i,
                        "source_role":state["case_role"],
                        "actual_visible_label":seen,
                        "physical_shape":"SolidSphere",
                        "inner_radius_m":0,
                        "outer_radius_m":shape["outer_radius_m"],
                        "actual_sigma_surface":"outer_only",
                        "named_source":"Sphere A",
                        "mode":"precomputed_static_not_particle_motion",
                        "screenshot":str(saved.relative_to(directory)),
                        "native_answer_values":{
                            key:expected for key,(expected,unit) in required.items()
                        },
                        "visible_readout_from_real_browser":True,
                        "actual_field_grid_resolution":[65,65],
                        "isolated_sigma_label":"Surface-charge density (one real outer face)",
                    })
                if external_requests:
                    raise AssertionError("offline source viewer made HTTP requests: "+str(external_requests))
                if js_errors:
                    raise AssertionError("browser JS error during actual solid rendering: "+str(js_errors))
            finally:
                page.close()
            # The M5 slider script must not silently repair invalid HTML
            # default visibility. Test the actual file in a JS-DISABLED browser.
            nojs_context=browser.new_context(java_script_enabled=False)
            try:
                nojs_page=nojs_context.new_page()
                try:
                    nojs_page.goto(filename.resolve().as_uri(),wait_until="load",timeout=45000)
                    fallback=nojs_page.locator("#m36e-source-observations [data-m36e-state]")
                    if fallback.count()!=3:
                        raise AssertionError("no-JS source fallback must contain all stored states")
                    fallback_visible=[i for i in range(3) if fallback.nth(i).is_visible()]
                    if fallback_visible!=[0]:
                        raise AssertionError(
                            "no-JS source fallback must show only original, not teacher worlds: "+
                            str(fallback_visible)
                        )
                    if "PRINTED ORIGINAL GEOMETRY" not in fallback.nth(0).inner_text():
                        raise AssertionError("no-JS source fallback omitted original actual answer")
                finally:
                    nojs_page.close()
            finally:
                nojs_context.close()
        finally:
            browser.close()
    # A co-forged manifest and HTML readout can agree but be nonphysical.
    # Preserve the distinct standard-library Gauss checker as an additional
    # mandatory defense instead of calling Chromium screenshot scientific proof.
    independent_physics=independently_check_native_sphere(manifest)
    report={
        "schema":"conductor_sim.m36d_real_chromium_solid.v1",
        "actual_chromium_source_and_teacher_states_checked":len(recorded),
        "original_source_solid_not_hollow":True,
        "genuine_offline_browser":True,
        "http_requests":0,
        "javascript_errors":0,
        "stored_static_states":recorded,
        "separate_physics_reference_required":True,
        "independent_source_native_SI_checks":independent_physics["actual_solid_source_states_independently_checked"],
        "actual_embedded_grid_SI_checked_points":grid_reference["actual_checked_potential_and_E_points"],
        "actual_js_disabled_original_only_fallback_verified":True,
    }
    (directory/"chromium-actual-solid.json").write_text(
        json.dumps(report,allow_nan=False,indent=2)+"\n",encoding="utf-8"
    )
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    r=check(args.output_dir)
    print("M36d REAL source SOLID original and teacher viewers inspected:",
          r["actual_chromium_source_and_teacher_states_checked"])
    print("M36d source-authentic disk and outer-only density verified")


if __name__=="__main__":
    main()
