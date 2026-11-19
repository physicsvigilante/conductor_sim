"""Actual JS-on/no-JS/mobile Chromium verification of teacher-only M39b.

Canonical HTML comes from real verified native M39a ledgers; visible text,
complete-face signs, provenance and CSS/HTML are independently inspected.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from conductor_sim.cavity_gauss import solve_cavity_gauss
from conductor_sim.cavity_topology_viewer import (
    render_cavity_topology, topology_view_manifest,
)
from conductor_sim.model import Grounded
from tools.m39a_gauss_example import sample_assembly
from tools.m39b_topology_reference import (
    WARNING, PROVENANCE, REFUSAL, check_topology_view_manifest,
)


def _signed(value, unit):
    if type(value) not in (float, int):
        raise AssertionError("signed SI must be real, not Boolean")
    v = float(value)
    return ("0 " + unit if v == 0.0 else
            ("+" if v > 0 else "−") + repr(abs(v)) + " " + unit)


def _expected():
    a = sample_assembly()
    isolated = solve_cavity_gauss(a)
    grounded = solve_cavity_gauss(replace(a, constraint=Grounded()))
    return (topology_view_manifest(isolated, grounded),
            render_cavity_topology(isolated, grounded))


def _verify_canonical(manifest, html_path):
    expected_manifest, html = _expected()
    def stable(x):
        return json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":"))
    if stable(manifest) != stable(expected_manifest):
        raise AssertionError("viewer manifest differs from genuine qualified source physics")
    if html_path.read_bytes() != html.encode("utf-8"):
        raise AssertionError("noncanonical visible HTML/CSS differs from genuine source renderer")


def _visible_text(page, selector, expected):
    loc = page.locator(selector)
    if loc.count() != 1 or not loc.is_visible() or loc.inner_text().strip() != expected:
        raise AssertionError("missing or false visible signed SI/provenance at " + selector)
    props = loc.evaluate("""el => {
        const s=getComputedStyle(el),r=el.getBoundingClientRect();
        return {display:s.display,visibility:s.visibility,opacity:Number(s.opacity),
                width:r.width,height:r.height};
    }""")
    if (props["display"] == "none" or props["visibility"] != "visible"
            or props["opacity"] <= 0 or props["width"] <= 0 or props["height"] <= 0):
        raise AssertionError("hidden actual signed/source value at " + selector)


def _check_visible(page, manifest, width):
    _visible_text(page, "#m39b-title", "Gauss law · multiple sealed cavities")
    _visible_text(page, "#m39b-warning", WARNING)
    _visible_text(page, "#m39b-membership", PROVENANCE)
    _visible_text(page, "#m39b-refusal", REFUSAL)
    _visible_text(page, "#m39b-metal", manifest["metal_id"])
    _visible_text(page, "#m39b-provenance",
                  "TEACHER HYPOTHETICAL · NO VERIFIED ORIGINAL JEE ANSWER")
    _visible_text(page, "#m39b-original-count",
                  "Original JEE source problems newly solved: 0. "
                  "Workbook p11 and irregular three-cavity notes p70 "
                  "remain unsupported for local 3D field/density.")
    _visible_text(page, "#m39b-zero-warning",
                  "A zero entire-face net charge does NOT imply zero "
                  "local surface charge density throughout that face.")
    if page.locator("script").count():
        raise AssertionError("the static topology viewer must not contain any JS")
    if page.locator("#m39b-cavities > .cavity").count() != len(manifest["cavities"]):
        raise AssertionError("visible cavity membership count differs from signed model")
    for idx, c in enumerate(manifest["cavities"]):
        _visible_text(page, f"#m39b-cavity-{idx}", c["cavity_id"])
        _visible_text(page, f"#m39b-q-{idx}", _signed(c["source_charge_sum_c"],"C"))
        _visible_text(page, f"#m39b-inner-{idx}",
                      _signed(c["induced_inner_face_total_charge_c"],"C"))
        for j, source in enumerate(c["authored_sources"]):
            row = page.locator(f"#m39b-source-{idx}-{j}")
            if row.count() != 1 or not row.is_visible():
                raise AssertionError("source ID or signed charge omitted from visible card")
            _visible_text(page, f"#m39b-source-{idx}-{j} code", source["source_id"])
            _visible_text(page, f"#m39b-source-{idx}-{j} .charge",
                          _signed(source["charge_c"],"C"))
        if not c["authored_sources"]:
            _visible_text(page, f"#m39b-source-empty-{idx}", "No sources assigned")
        box=page.locator(f"#m39b-card-{idx}").bounding_box()
        if not box or box["width"] < 160 or box["x"] < -1 or (box["x"]+box["width"]) > width + 2:
            raise AssertionError("mobile viewport clipped a physical source/cavity identity")
    for case in ("isolated","grounded"):
        state=manifest["states"][case]
        _visible_text(page,f"#m39b-metal-{case}",
                      _signed(state["conductor"]["equilibrium_net_metal_charge_c"],"C"))
        _visible_text(page,f"#m39b-outer-{case}",_signed(state["outer_face_total_charge_c"],"C"))
        _visible_text(page,f"#m39b-flux-{case}",_signed(state["exterior_electric_flux_v_m"],"V·m"))
        _visible_text(page,f"#m39b-sources-{case}",
                      _signed(state["total_interior_source_charge_c"],"C"))
    _visible_text(page,"#m39b-fixed-isolated",
                  _signed(manifest["states"]["isolated"]["conductor"]["authored_isolated_metal_charge_c"],"C"))
    _visible_text(page,"#m39b-fixed-grounded","Not a fixed-charge input")
    _visible_text(page,"#m39b-theorem-isolated","Not generally implied for isolated metal")
    _visible_text(page,"#m39b-theorem-grounded",
                  "Yes, under declared no-exterior-source boundary")
    if not page.locator("#m39b-summary").is_visible():
        raise AssertionError("signed complete-face comparison table invisible")
    if width <= 460:
        scroll=page.locator(".table-scroll").evaluate(
            "el => ({client:el.clientWidth,scroll:el.scrollWidth})"
        )
        if scroll["scroll"] <= scroll["client"]:
            raise AssertionError("mobile SI comparison table does not expose horizontal scrolling")


def _attack_html(manifest, html_path, folder):
    folder.mkdir(parents=True,exist_ok=True)
    original=html_path.read_text(encoding="utf-8")
    attacks=(
        ("hide-all-text","color:#193149","color:transparent"),
        ("pseudo-fake","</style>","body:before{content:'FORGED';display:block}</style>"),
        ("fake-source","TEACHER HYPOTHETICAL","ORIGINAL JEE SOLVED"),
        ("fake-face-sign",'id="m39b-inner-1">+1e-09 C','id="m39b-inner-1">−1e-09 C'),
        ("fake-grounding","Yes, under declared no-exterior-source boundary",
                           "Yes, regardless of external sources"),
        ("fake-coverage","Original JEE source problems newly solved: ",
                         "Original JEE source problems newly solved: 7 "),
    )
    for name,before,after in attacks:
        if before not in original:
            raise AssertionError("hostile test has no authentic match for " + name)
        fake=folder/(name+".html")
        fake.write_text(original.replace(before,after,1),encoding="utf-8")
        try:
            _verify_canonical(manifest,fake)
        except AssertionError as ex:
            if "noncanonical visible HTML/CSS" not in str(ex):
                raise
        else:
            raise AssertionError("forged HTML/CSS passed source integrity: "+name)


def check(folder):
    from playwright.sync_api import sync_playwright
    html_path=folder/"m39b-topology-view.html"
    manifest=json.loads((folder/"m39b-topology-view.json").read_text(encoding="utf-8"))
    witnessed=check_topology_view_manifest(manifest)
    if witnessed["original_workbook_items_newly_solved"] != 0:
        raise AssertionError("falsified original workbook source attribution")
    _verify_canonical(manifest,html_path)
    _attack_html(manifest,html_path,folder/"hostile-html")
    screenshots=folder/"screenshots"
    screenshots.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        try:
            for js_off in (False,True):
                for width,height in ((1280,900),(375,900)):
                    page=browser.new_page(
                        viewport={"width":width,"height":height},
                        java_script_enabled=not js_off,device_scale_factor=1
                    )
                    requests, errors = [], []
                    page.on("request",lambda req: requests.append(req.url)
                            if req.url.startswith(("http://","https://")) else None)
                    page.on("pageerror",lambda ex: errors.append(str(ex)))
                    page.goto(html_path.resolve().as_uri(),wait_until="load",timeout=45000)
                    _check_visible(page,manifest,width)
                    if requests or errors:
                        raise AssertionError("offline static page made HTTP(S) requests or raised JS errors")
                    suffix=("nojs" if js_off else "js")+"-"+str(width)
                    page.screenshot(path=str(screenshots/(suffix+".png")),full_page=True)
                    page.close()
        finally:
            browser.close()
    return {"actual_chromium_cases":4,"hostile_html_refusals":6,
            "cavities":len(manifest["cavities"]),
            "newly_solved_original_jee_items":0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=Path("output/m39b"))
    args=p.parse_args()
    print(json.dumps(check(args.output_dir),sort_keys=True))


if __name__=="__main__":
    main()
