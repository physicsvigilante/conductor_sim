"""M38b genuine Chromium HTML-only hostile first-frame proof.

Keep original native SI JSON AND embedded original JSON unchanged for every
attack. Verify forgery ACTUALLY visible with JavaScript disabled, then require
actual production Chromium checker to refuse for the intended physical reason.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
import sys

from tools.check_m38b_line_browser import check as production_chromium
from tools.m38b_line_view_reference import check_line_view


# label, HTML-only regex (group 1 kept), false value, actual DOM selector,
# DOM evidence mode, DOM attribute, intended production refusal substring.
ATTACKS = (
    (
        "charge-x-first", r'(<circle id="m38b-charge" cx=")[^"]+', "599",
        "#m38b-charge", "attribute", "cx",
        "no-JS first physical scene/current marker mismatch"
    ),
    (
        "charge-y-first", r'(<circle id="m38b-charge" cx="[^"]+" cy=")[^"]+', "285",
        "#m38b-charge", "attribute", "cy",
        "no-JS first physical scene/current marker mismatch"
    ),
    (
        "charge-sign", r'(<text id="m38b-q-sign"[^>]*>)[^<]+', "−q",
        "#m38b-q-sign", "text", None,
        "visible source charge sign contradicts physical signed q"
    ),
    (
        "sphere-radius", r'(<circle id="m38b-sphere" cx="[^"]+" cy="[^"]+" r=")[^"]+', "13",
        "#m38b-sphere", "attribute", "r",
        "false visible fixed physical sphere geometry"
    ),
    (
        "sphere-centre-x", r'(<circle id="m38b-sphere" cx=")[^"]+', "355",
        "#m38b-sphere", "attribute", "cx",
        "false visible fixed physical sphere geometry"
    ),
    (
        "path-end-x", r'(<line id="m38b-path"[^>]* x2=")[^"]+', "558",
        "#m38b-path", "attribute", "x2",
        "false visible complete 3D projected source path"
    ),
    (
        "plotted-current-first-point", r'(<polyline id="m38b-current-line" points=")[^ ]+', "0,0",
        "#m38b-current-line", "attribute-prefix", "points",
        "visible plotted current point 0 contradicts verified Kelvin SI"
    ),
    (
        "first-current-marker-x", r'(<circle id="m38b-current-marker" cx=")[^"]+', "450",
        "#m38b-current-marker", "attribute", "cx",
        "no-JS first physical scene/current marker mismatch"
    ),
    (
        "first-current-marker-y", r'(<circle id="m38b-current-marker" cx="[^"]+" cy=")[^"]+', "245",
        "#m38b-current-marker", "attribute", "cy",
        "no-JS first physical scene/current marker mismatch"
    ),
    (
        "false-initial-current", r'(<strong class="value" id="m38b-current">)[^<]+', "-9 A",
        "#m38b-current", "text", None,
        "no-JS first SI/world label mismatch"
    ),
    (
        "false-opposite-current", r'(<strong class="value" id="m38b-opposite">)[^<]+', "123 A",
        "#m38b-opposite", "text", None,
        "no-JS first SI/world label mismatch"
    ),
    (
        "false-initial-time", r'(<strong id="m38b-time">)[^<]+', "999 s",
        "#m38b-time", "text", None,
        "no-JS first SI/world label mismatch"
    ),
    (
        "false-initial-distance", r'(<strong class="value" id="m38b-distance">)[^<]+', "333 m",
        "#m38b-distance", "text", None,
        "no-JS first SI/world label mismatch"
    ),
    (
        "false-initial-world-xyz", r'(<strong class="value" id="m38b-world">)[^<]+', "[99, 0, 0] m",
        "#m38b-world", "text", None,
        "no-JS first SI/world label mismatch"
    ),
    (
        "false-initial-radial-speed", r'(<strong class="value" id="m38b-radial-v">)[^<]+', "50 m/s",
        "#m38b-radial-v", "text", None,
        "no-JS first SI/world label mismatch"
    ),
    (
        "false-source-attribution", r'(<h1 id="m38b-heading">)[^<]+', "PRINTED WORKBOOK Q21 EXACT OFF-AXIS MOTION",
        "#m38b-heading", "text", None,
        "false visible teacher-vs-original source attribution"
    ),
    (
        "false-initial-slider-index", r'(<input id="m38b-index"[^>]* value=")[^"]+', "15",
        "#m38b-index", "attribute", "value",
        "no-JS first physical time index falsely selected"
    ),
    (
        "falsified-current-sign-axis",
        r'(<text x="60" y="17"[^>]*>)[^<]+', "I (A), negative sphere → ground",
        "#m38b-current-chart text", "chart-heading", None,
        "visible current/physical-time axis label falsified"
    ),
    (
        "falsified-local-physical-u-axis",
        r'(<text x="676" y="181"[^>]*>)[^<]+', "+world-x",
        "#m38b-scene text", "scene-u", None,
        "visible local source physical axes falsified"
    ),
    (
        "falsely-enabled-controls", r'(<input id="m38b-index"[^>]* value="0") disabled', "",
        "#m38b-index", "enabled", None,
        "no-JS first physical time controls must remain disabled"
    ),
)


def check_hostiles(directory: Path):
    from playwright.sync_api import sync_playwright

    page=directory/"m38b-line.html"
    genuine=page.read_bytes()
    genuine_text=genuine.decode("utf-8")
    native=(directory/"m38b-line-physics.json").read_bytes()
    manifest=(directory/"m38b-line-viewer.json").read_bytes()
    oracle=check_line_view(json.loads(manifest))
    if oracle["independently_checked_kelvin_states"] != 41:
        raise AssertionError("expected genuine 41-sample baseline physics")
    if not genuine_text.startswith("<!doctype html>"):
        raise AssertionError("invalid real M38b HTML baseline")
    screenshots=directory/"hostile-screenshots"
    screenshots.mkdir(parents=True,exist_ok=True)
    outcomes=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        context=browser.new_context(java_script_enabled=False,
                                    viewport={"width":960,"height":790})
        try:
            visible=context.new_page()
            for attack, pattern, forged_value, selector, kind, attr, error in ATTACKS:
                fake, count = re.subn(pattern,lambda match: match.group(1)+forged_value,
                                      genuine_text,count=1)
                if count!=1 or fake==genuine_text or '<script id="m38b-data"' not in fake:
                    raise AssertionError("hostile initial HTML-only attack not constructed: "+attack)
                page.write_text(fake,encoding="utf-8")
                try:
                    if ((directory/"m38b-line-physics.json").read_bytes()!=native or
                            (directory/"m38b-line-viewer.json").read_bytes()!=manifest):
                        raise AssertionError("original native independent Kelvin SI/manifest changed")
                    visible.goto(page.resolve().as_uri(),wait_until="load",timeout=45000)
                    loc=visible.locator(selector)
                    # A horizontal SVG <line> has a zero-height geometric
                    # bounding box even though its positive-width SVG stroke
                    # is genuinely visible. Playwright is_visible() alone
                    # falsely reports hidden for that one legitimate path.
                    stroked_line=(selector=="#m38b-path" and loc.evaluate(
                        "(el)=>{const s=getComputedStyle(el),r=el.ownerSVGElement;"
                        "return !!r&&r.getBoundingClientRect().width>0&&"
                        "s.display!=='none'&&s.visibility!=='hidden'&&"
                        "Number(s.opacity)>0&&s.stroke!=='none'&&"
                        "parseFloat(s.strokeWidth)>0&&"
                        "Number(el.getAttribute('x1'))!==Number(el.getAttribute('x2'));}"
                    ))
                    if not loc.is_visible() and not stroked_line:
                        raise AssertionError("fake no-JS physical datum not actually visible: "+attack)
                    if kind=="text":
                        # SVG text nodes are not HTMLElements; Playwright
                        # inner_text fails on them. text_content works for
                        # both genuine SVG and HTML first-frame claims.
                        observed=(loc.text_content() or "").strip()
                        if observed!=forged_value:
                            raise AssertionError("forged no-JS scientific text not actually visible: "+attack)
                    elif kind=="attribute":
                        if loc.get_attribute(attr)!=forged_value:
                            raise AssertionError("forged no-JS physical SVG attribute absent: "+attack)
                    elif kind=="attribute-prefix":
                        if not (loc.get_attribute(attr) or "").startswith(forged_value+" "):
                            raise AssertionError("forged current plot vertex not actually visible: "+attack)
                    elif kind=="chart-heading":
                        # SVG text is genuinely rendered; the first text node
                        # in the signed-current SVG is the scientific I axis.
                        if visible.locator("#m38b-current-chart text").first.text_content().strip()!=forged_value:
                            raise AssertionError("forged current sign axis not visible: "+attack)
                    elif kind=="scene-u":
                        # Scene text order: charge sign, Grounded, +u, +w.
                        if visible.locator("#m38b-scene text").nth(2).text_content().strip()!=forged_value:
                            raise AssertionError("forged local u axis not visible: "+attack)
                    elif kind=="enabled":
                        if loc.is_disabled():
                            raise AssertionError("forged no-JS control is still disabled: "+attack)
                    else:
                        raise AssertionError("unknown hostile proof mode")
                    visible.screenshot(path=str(screenshots/(attack+".png")),full_page=True)
                    # Playwright Sync API cannot be nested within its running
                    # event loop. Execute the *unchanged actual production*
                    # Chromium gate in a separate Python process, preserving
                    # its real nonzero exit code and first error message.
                    result=subprocess.run(
                        [sys.executable,"-m","tools.check_m38b_line_browser",
                         "--output-dir",str(directory)],
                        capture_output=True,text=True,timeout=160,check=False
                    )
                    if result.returncode==0:
                        raise AssertionError(attack+": actual Chromium accepted false static original physics")
                    reason=result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
                    if not reason.startswith("AssertionError:") or error not in reason:
                        raise AssertionError(
                            attack+": WRONG actual browser scientific refusal: "+repr(reason)
                        )
                    outcomes.append({
                        "attack":attack,"no_js_forgery_visibly_proved":True,
                        "refusal":"REJECTED_AS_INTENDED","reason":reason
                    })
                finally:
                    page.write_bytes(genuine)
        finally:
            context.close()
            browser.close()
    if ((directory/"m38b-line-physics.json").read_bytes()!=native or
            (directory/"m38b-line-viewer.json").read_bytes()!=manifest or page.read_bytes()!=genuine):
        raise AssertionError("hostility runner did not restore genuine native/HTML bytes")
    if len(outcomes)!=len(ATTACKS):
        raise AssertionError("genuine hostile HTML-only scientific attacks incomplete")
    final=production_chromium(directory)
    if not final["javascript_disabled_first_frame"]:
        raise AssertionError("original no-JavaScript real baseline not restored")
    answer={
        "schema":"conductor_sim.m38b_html_only_hostility.v1",
        "actual_chromium_attacks":len(outcomes),
        "physically_refused":len(outcomes),
        "unexpected_acceptances":0,
        "genuine_html_byte_exact_restored":True,
        "native_si_unchanged":True,
        "independent_physics_kept":oracle["independently_checked_kelvin_states"],
        "cases":outcomes
    }
    (directory/"m38b-hostile-browser-report.json").write_text(
        json.dumps(answer,indent=2)+"\n",encoding="utf-8"
    )
    return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m38b"))
    args=parser.parse_args()
    report=check_hostiles(args.output_dir)
    print(json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True))


if __name__=="__main__":
    main()
