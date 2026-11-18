"""M36d: inspect REAL source-authentic SOLID conductor HTML in offline Chromium.

The numerical Gauss-law reference belongs to an entirely separate process.
This script verifies actual source/teacher static-state visibility and genuine
SOLID disk geometry on the viewer's *embedded* live frame data, no network.
"""
from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path


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
                for phrase in (
                    "genuinely SOLID conducting sphere",
                    "NOT the historical hollow-shell proxy",
                    "SYMBOLIC",
                    "TEACHER",
                    "PRECOMPUTED STATIC",
                    "NO physical inner face",
                ):
                    if phrase.casefold() not in banner_text.casefold():
                        raise AssertionError("missing actual visible truthful source label: "+phrase)
                if page.locator("canvas#scene").count()!=1:
                    raise AssertionError("no genuine painted physical conductor canvas")
                answer_panel=page.locator("#m36e-source-observations")
                if not answer_panel.is_visible():
                    raise AssertionError("JEE original source potential readouts not visibly presented")
                if "Original JEE p26: native potential observations" not in answer_panel.inner_text():
                    raise AssertionError("original textbook question not visibly identified")
                if "SYMBOLIC" not in answer_panel.inner_text() or "teacher-chosen" not in answer_panel.inner_text():
                    raise AssertionError("source-given symbols and teacher numbers not distinguished")
                if page.locator("#density-panel h2").inner_text()!="Surface-charge density (one real outer face)":
                    raise AssertionError("isolated fixed-Q sigma mislabeled induced charge")
                if page.locator("#profile").get_attribute("aria-label")!="Surface-charge density along a great circle":
                    raise AssertionError("density chart accessible label is physically misleading")
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
        finally:
            browser.close()
    report={
        "schema":"conductor_sim.m36d_real_chromium_solid.v1",
        "actual_chromium_source_and_teacher_states_checked":len(recorded),
        "original_source_solid_not_hollow":True,
        "genuine_offline_browser":True,
        "http_requests":0,
        "javascript_errors":0,
        "stored_static_states":recorded,
        "separate_physics_reference_required":True,
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
