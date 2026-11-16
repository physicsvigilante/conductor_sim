"""M28a real Chromium: offline saved-time trajectory sweep and censoring gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from conductor_sim import (FixedPointField, PointCharge, PrescribedField,
                           TestChargeMotionSpec, UniformElectricField,
                           SweepAxis, TrajectorySweepPlan,
                           run_trajectory_sweep, write_trajectory_sweep_html)
from conductor_sim.scenarios import PrescribedScenario
from examples.m28a_ring_sweep import run_case
from examples.m28a_guard_transfer import run_case as gravity_transfer_case


def guard_case():
    baseline = PrescribedScenario(
        "m28:guard-browser",
        TestChargeMotionSpec(
            PrescribedField((FixedPointField(-1e-10, (.2, 0., 0.)),)),
            PointCharge(1e-9, (0., 0., 0.)), 1e-4, (0., 0., 0.),
            .25, .01, plane_normal=(0., 0., 1.), source_guard_m=.005),
        (("point:held", "Held negative point"),),
        "Teacher hypothetical to show source guard; no contact dynamics.")
    return run_trajectory_sweep(TrajectorySweepPlan(
        "m28:guard-browser-sweep", baseline,
        SweepAxis("test-particle", "initial_velocity_m_s", ((1., 0., 0.),))))



def uniform_case():
    baseline = PrescribedScenario(
        "m28:uniform-browser",
        TestChargeMotionSpec(
            PrescribedField((UniformElectricField((40., 0., 0.)),)),
            PointCharge(1e-9, (0., 0., 0.)),
            1e-5, (0., 0., 0.), .05, .01,
            plane_normal=(0., 0., 1.)),
        (("field:uniform", "Held uniform electric field"),),
        "Teacher hypothetical uniform field, unrelated to ring workbook.")
    return run_trajectory_sweep(TrajectorySweepPlan(
        "m28:uniform-browser-sweep", baseline,
        SweepAxis("field:uniform", "field_v_m", ((55., 0., 0.),))))


def check(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    errors, consoles, requests = [], [], []
    results = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for kind in ("ring-b-charge", "ring-b-separation", "guard",
                     "uniform-only", "baseline-guard"):
            if kind == "guard":
                sweep = guard_case()
            elif kind == "uniform-only":
                sweep = uniform_case()
            elif kind == "baseline-guard":
                sweep = gravity_transfer_case()
            else:
                sweep = run_case(kind)
            html = write_trajectory_sweep_html(sweep, output/(kind+".html"))
            page = browser.new_page(viewport={"width": 1360, "height": 900})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda e: consoles.append(e.text) if e.type == "error" else None)
            page.on("request", lambda r: requests.append(r.url))
            page.set_content(html.read_text(encoding="utf-8"), wait_until="load")
            page.wait_for_function("window.m28SweepViewer && window.m28SweepViewer.loaded")
            data = page.evaluate("""() => JSON.parse(
                document.getElementById('m28-sweep-data').textContent)""")
            assert data["schema"] == "conductor_sim.trajectory_sweep_view.v1"
            assert data["browser_ode"] is False and data["time_interpolation"] is False
            assert data["frame_times_s"][0] == 0
            assert page.locator("#sweep-plot polyline").count() == len(
                [x for x in data["tracks"] if x["samples"]])
            assert page.evaluate("window.m28SweepViewer.physicalTime") == 0
            assert page.locator("#sweep-states tr").count() == len(data["tracks"])
            assert "not an apparatus/ring-geometry drawing" in page.locator("body").inner_text().lower()
            visible = page.locator("body").inner_text()
            assert "Authored baseline" in visible
            assert "not independently source-verified" in visible
            assert data["baseline_provenance"] == sweep.plan.baseline.provenance
            assert data["baseline_origin"] == sweep.plan.baseline.origin
            if kind.startswith("ring-b-"):
                assert "Workbook physical p12" in visible
                assert [t["status"] for t in data["tracks"]] == [
                    "complete", "complete", "complete"]
                assert data["source_ids"] == ["ring:A", "ring:B", "field:up"]
            else:
                # The generic renderer MUST NOT misattribute a point-only
                # or uniform-only apparatus to workbook ring questions.
                assert "Workbook" not in visible and "Q40" not in visible
                assert "probe q=+10" not in visible
                if kind == "guard":
                    assert sweep.entries[0].status == "censored"
                    assert data["tracks"][1]["termination"] == "source-guard"
                    assert data["source_ids"] == ["point:held"]
                elif kind == "uniform-only":
                    assert data["source_ids"] == ["field:uniform"]
                    assert [t["status"] for t in data["tracks"]] == [
                        "complete", "complete"]
                else:
                    assert data["source_ids"] == ["point:review-held"]
                    assert sweep.baseline_trajectory.termination == "source-guard"
                    assert sweep.entries[0].status == "complete"
                    assert data["tracks"][0]["termination"] == "source-guard"
                    assert data["tracks"][1]["termination"] == "duration"
                    assert data["frame_times_s"][-1] == .30
            # Prove the play control really steps saved states.
            page.locator("#play").click()
            page.wait_for_function("window.m28SweepViewer.frameIndex > 0", timeout=5000)
            page.locator("#play").click()
            # Jump to the exact last baseline saved state; *no interpolation*.
            page.locator("#frame").evaluate("""(el) => {
                el.value=el.max;
                el.dispatchEvent(new Event('input',{bubbles:true}));
            }""")
            end = page.evaluate("window.m28SweepViewer.physicalTime")
            assert end == data["frame_times_s"][-1]
            active = page.evaluate("window.m28SweepViewer.activeCount")
            missing = page.evaluate("window.m28SweepViewer.missingCount")
            if kind in ("guard", "baseline-guard"):
                assert active == 1 and missing == 1
                assert "No saved state" in page.locator("#sweep-states").inner_text()
                # Neither a guarded child nor an early-censored baseline
                # can invent a physical state at another track's later time.
                missing_marker = 1 if kind == "guard" else 0
                assert page.locator("#sweep-plot circle").nth(missing_marker).evaluate(
                    "(el) => getComputedStyle(el).display") == "none"
                if kind == "baseline-guard":
                    assert "Reference unavailable at this time" in page.locator(
                        "#sweep-states").inner_text()
                    assert "No exact-time comparison" in page.locator(
                        "#sweep-states").inner_text()
            else:
                assert active == (2 if kind == "uniform-only" else 3) and missing == 0
                assert "|Δr|" in page.locator("#sweep-states").inner_text()
            page.screenshot(path=str(output/(kind+"-wide.png")), full_page=True)
            page.locator("#reset").click()
            assert page.evaluate("window.m28SweepViewer.physicalTime") == 0
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(output/(kind+"-mobile.png")), full_page=True)
            mobile = page.evaluate("""() => ({
                viewport:window.innerWidth,
                document:document.documentElement.scrollWidth,
                svg:document.getElementById('sweep-plot').getBoundingClientRect().width
            })""")
            assert mobile["viewport"] == 390
            if mobile["document"] > 390:
                offenders = page.evaluate("""() => [...document.querySelectorAll('*')]
                  .map(el => ({tag:el.tagName,id:el.id,
                    right:el.getBoundingClientRect().right,
                    width:el.getBoundingClientRect().width}))
                  .filter(x => x.right > window.innerWidth+1)
                  .slice(0,12)""")
                raise AssertionError((mobile, offenders))
            assert mobile["svg"] <= 390, mobile
            results[kind] = dict(status="pass", source_ids=data["source_ids"],
                                 baseline_provenance=data["baseline_provenance"],
                                 saved_states=len(data["frame_times_s"]),
                                 final_saved_physical_time_s=end,
                                 active_at_end=active, unavailable_at_end=missing,
                                 mobile=mobile)
            page.close()
        browser.close()
    assert not errors and not consoles and not requests, (errors, consoles, requests)
    report = dict(status="pass", cases=results, page_errors=errors,
                  console_errors=consoles, network_requests=requests,
                  facts="Python physical samples; no browser ODE or time interpolation",
                  provenance="Case-specific authored baseline metadata. Workbook ring and "
                             "probe claims apply only to ring cases; all variation "
                             "parameters are teacher hypotheses.")
    (output/"m28a-browser.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/m28a-browser"))
    args = parser.parse_args()
    print(json.dumps(check(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
