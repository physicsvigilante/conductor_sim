"""M37b: real Chromium original HTML-only no-JS evidence attacks.

Independently verified JSON and the embedded JSON remain genuine. For each
mutation only the literal initial SVG/HTML claim changes; JS-enabled startup
would rewrite it, while JS-off original first-frame viewers would lie.

Execute the ACTUAL production browser gate for each altered document.
All intended failures must identify the no-JS scientific condition.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tools.check_m37b_prescribed_browser import check
from tools.m37b_source_reference import check as check_si


ATTACKS = (
    ("initial-marker-x", r'(<circle id="m37b-current-marker" cx=")[^"]+', "500",
     "no-JS physical scene/graph coordinate #m37b-current-marker.cx"),
    ("initial-marker-y", r'(<circle id="m37b-current-marker" cx="[^"]+" cy=")[^"]+', "230",
     "no-JS physical scene/graph coordinate #m37b-current-marker.cy"),
    ("initial-charge-x", r'(<circle id="m37b-charge" cx=")[^"]+', "690",
     "no-JS physical scene/graph coordinate #m37b-charge.cx"),
    ("initial-charge-sign-x", r'(<text id="m37b-q-sign" x=")[^"]+', "690",
     "no-JS physical scene/graph coordinate #m37b-q-sign.x"),
    ("false-initial-time", r'(<strong id="m37b-time">)[^<]+', "999 s",
     "no-JS first physical SI label #m37b-time"),
    ("false-initial-distance", r'(<strong class="value" id="m37b-distance">)[^<]+', "333 m",
     "no-JS first physical SI label #m37b-distance"),
    ("false-initial-opposite-current",
     r'(<strong class="value" id="m37b-opposite">)[^<]+', "+9 A",
     "no-JS first physical SI label #m37b-opposite"),
    ("false-initial-charge-sign", r'(<text id="m37b-q-sign"[^>]*>)[^<]+', "−q",
     "no-JS source charge sign label"),
    ("false-initial-sphere-radius", r'(<circle id="m37b-sphere"[^>]* r=")[^"]+', "7",
     "no-JS physical scene/graph coordinate #m37b-sphere.r"),
    ("false-initial-slider-time", r'(<input id="m37b-index"[^>]* value=")[^"]+', "15",
     "no-JS first physical time index"),
)


def check_hostiles(directory: Path):
    page=directory/"p8-ground-current.html"
    genuine=page.read_text(encoding="utf-8")
    source_json=(directory/"p8-ground-current.json").read_bytes()
    trusted=json.loads(source_json)
    oracle=check_si(trusted)
    if oracle["verified_python_kelvin_states"]!=31:
        raise AssertionError("expected real 31-state M37b source evidence")
    if not genuine.startswith("<!doctype html>"):
        raise AssertionError("not the legitimate source-generated offline HTML")
    outcomes=[]
    try:
        for label,pattern,forgery,required_error in ATTACKS:
            forged,count=re.subn(pattern,lambda m: m.group(1)+forgery,
                                 genuine,count=1)
            if count!=1 or forged==genuine or "<script id=\"m37b-data\"" not in forged:
                raise AssertionError(f"hostile setup failed: {label}")
            page.write_text(forged,encoding="utf-8")
            if (directory/"p8-ground-current.json").read_bytes()!=source_json:
                raise AssertionError("the science JSON changed during HTML-only hostility")
            try:
                check(directory)
            except AssertionError as error:
                if required_error not in str(error):
                    raise AssertionError(
                        f"{label}: wrong browser proof refusal {error!r}"
                    ) from error
                outcomes.append({
                    "attack":label,
                    "outcome":"REJECTED_AS_INTENDED",
                    "scientific_reason":str(error),
                })
            else:
                raise AssertionError(f"{label}: REAL Chromium browser gate accepted false original no-JS physics")
    finally:
        page.write_text(genuine,encoding="utf-8")
    if (directory/"p8-ground-current.json").read_bytes()!=source_json:
        raise AssertionError("genuine Kelvin output changed after adversarial checks")
    if page.read_text(encoding="utf-8")!=genuine:
        raise AssertionError("genuine HTML was not restored exactly")
    if len(outcomes)!=len(ATTACKS):
        raise AssertionError("not all independent no-JS attacks were executed")
    result={"schema":"m37b.nojs_html_adversarial.v1",
            "original_source_examples":1,
            "attacks":len(outcomes),
            "refused":len(outcomes),
            "unexpected_acceptances":0,
            "genuine_html_restored":True,
            "science_json_unchanged":True,
            "outcomes":outcomes}
    (directory/"nojs-adversarial-browser.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m37b"))
    args=parser.parse_args()
    report=check_hostiles(args.output_dir)
    print(json.dumps({k:v for k,v in report.items() if k!="outcomes"},sort_keys=True))


if __name__=="__main__":
    main()
