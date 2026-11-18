"""M36e: real Chromium must refuse forged textbook-answer presentation evidence.

This is a separate browser gate (not generic Python tests) and intentionally
edits copies of the REAL exported HTML/native JSON. A green screenshot with
fabricated potential values or a stale source/teacher state MUST fail.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from tools.check_m36d_solid_browser import check


def check_hostile(directory:Path)->dict:
    raw=(directory/"native-solid-sphere.json").read_text(encoding="utf-8")
    manifest=json.loads(raw)
    viewer=manifest["viewer_file"]
    source_html=(directory/viewer).read_text(encoding="utf-8")
    original=manifest["native_states"][0]["native_samples"]["half_radius_potential_v"]
    precise=format(original,".17g")
    visible=format(original,".11g")

    def substitution(old,new):
        if source_html.count(old)!=1:
            raise AssertionError(
                f"expected exactly one authentic HTML source marker before tampering: {old[:60]!r}"
            )
        return source_html.replace(old,new,1)

    number_prefix='data-native-value="'+precise+'">'+visible+' V</strong>'
    altered_visible=substitution(
        number_prefix,
        'data-native-value="'+precise+'">999.25 V</strong>',
    )
    tamper_cases=[
        ("forged visible original V(R/2)",altered_visible,raw,
         "visible original source answer is stale/forged"),
        ("forged numeric attribute of original V(R/2)",
         substitution('data-native-value="'+precise+'"',
                      'data-native-value="999.25"'),raw,
         "display-native source evidence mismatch"),
        ("duplicated original-source card index",
         substitution('data-m36e-state="1"','data-m36e-state="0"'),raw,
         "visible source answer card is stale"),
        ("teacher card falsely attributed to printed original",
         substitution('data-m36e-role="teacher_changed_radius_charge_and_centre"',
                      'data-m36e-role="source_original_shape_teacher_numeric_example"'),raw,
         "answer card mistaken original textbook"),
        ("isolated net charge mislabelled as induced",
         substitution("<h2>Surface-charge density (one real outer face)</h2>",
                      "<h2>Induced surface-charge profile</h2>"),raw,
         "isolated fixed-Q sigma mislabeled"),
        ("falsely removed source-symbolic warning",
         substitution("The printed source supplies SYMBOLIC R and Q",
                      "The printed source supplies NUMERIC R and Q"),raw,
         "source-given symbols and teacher numbers not distinguished"),
    ]
    altered_native=deepcopy(manifest)
    altered_native["native_states"][0]["native_samples"]["half_radius_potential_v"]=42.
    tamper_cases.append((
        "native manifest potential changed but answer HTML left stale",
        source_html,json.dumps(altered_native),
        "display-native source evidence mismatch",
    ))
    accepted=[]
    for name,html,text,expected_error in tamper_cases:
        with tempfile.TemporaryDirectory(prefix="m36e-browser-adversary-") as scratch:
            folder=Path(scratch)
            (folder/viewer).write_text(html,encoding="utf-8")
            (folder/"native-solid-sphere.json").write_text(text,encoding="utf-8")
            try:
                check(folder)  # Real Chromium, real slider, actual DOM.
            except AssertionError as exc:
                if expected_error not in str(exc):
                    raise AssertionError(
                        f"{name}: rejected for wrong reason: {exc}"
                    ) from exc
                accepted.append({"forgery":name,"result":"EXPECTED_REFUSAL",
                                 "reason":str(exc)})
            else:
                raise AssertionError("real Chromium accepted forged source teaching evidence: "+name)
    report={
        "schema":"conductor_sim.m36e_adversarial_readout_browser.v1",
        "genuine_source_html_and_native_input":True,
        "real_browser_hostile_forgery_count":len(accepted),
        "real_browser_expected_refusals":accepted,
        "original_source_apparatus_added":0,
    }
    (directory/"adversarial-browser-readout.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    report=check_hostile(args.output_dir)
    print("M36e REAL Chromium textbook-readout adversarial refusals:",
          report["real_browser_hostile_forgery_count"])


if __name__=="__main__":
    main()
