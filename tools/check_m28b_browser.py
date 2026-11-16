"""M28b offline Chromium: authentic saved event roots, provenance and mobile view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from examples.m28b_event_sweep import CASES, run_case
from conductor_sim.event_sweep_viewer import write_event_sweep_html


def check(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    failures, consoles, requests = [], [], []
    cases = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for case in CASES:
            run = run_case(case)
            path = write_event_sweep_html(
                run, output/(case+".html"),
                title="M28b · Native event sweep · "+case)
            json_path=path.with_suffix(".json")
            json_path.write_text(json.dumps(run.as_dict(),indent=2,allow_nan=False)+"\n",
                                 encoding="utf-8")
            page = browser.new_page(viewport={"width":1360,"height":900})
            page.on("pageerror", lambda e: failures.append(str(e)))
            page.on("console", lambda e: consoles.append(e.text) if e.type=="error" else None)
            page.on("request", lambda r: requests.append(r.url))
            page.set_content(path.read_text(encoding="utf-8"), wait_until="load")
            evidence = page.evaluate("""() => JSON.parse(
                document.getElementById('m28b-evidence').textContent)""")
            assert evidence["schema"] == "conductor_sim.event_sweep.v1"
            assert evidence["time_interpolation"] is False
            assert evidence["postguard_extrapolation"] is False
            assert evidence["baseline_revision_id"]==run.plan.sweep.baseline.revision_id
            assert page.locator("#provenance").count()==1
            assert run.plan.sweep.baseline.provenance in page.locator("#provenance").inner_text()
            assert "unverified" in page.locator("#provenance").inner_text()
            assert page.locator("#event-timeline").count()==1
            assert "no browser ODE" in page.locator("body").inner_text()
            assert page.locator("#event-rows tr").count() == (
                len(run.plan.questions)*(len(run.entries)+1))
            actual_roots=len(run.baseline_run.events)+sum(
                len(e.run.events) for e in run.entries if e.run is not None)
            assert page.locator("#event-timeline circle").count() == actual_roots
            assert [x["requested_value"] for x in evidence["entries"]]==[
                x["requested_value"] for x in run.as_dict()["entries"]]
            assert all(e["time_interpolation"] is False for e in
                       ({"time_interpolation":evidence["time_interpolation"]},))
            if case=="ring-detector":
                assert evidence["axis"]["component_id"]=="ring:B"
                assert "ring:A" in page.locator("#source-ids").inner_text()
                assert "Workbook" in page.locator("#provenance").inner_text()
            else:
                assert "Workbook" not in page.locator("#provenance").inner_text()
                assert "Q40" not in page.locator("#provenance").inner_text()
            if case=="uniform-mass":
                comparisons=evidence["entries"][0]["event_time_comparisons"]
                assert comparisons[0]["status"]=="one-resolved-pair"
                assert len(comparisons[0]["numerical_bracket_shift_interval_s"])==2
            if case=="multi-crossing":
                assert evidence["entries"][0]["event_time_comparisons"][0]["status"]=="unpaired"
                assert page.locator("#event-timeline circle").count()==4
            if case=="coincident-terminal":
                native = run.baseline_run
                assert native.trajectory.termination == "terminal-event"
                assert len(native.events) == 2
                assert any(e.sample.time_s > native.trajectory.samples[-1].time_s
                           for e in native.events)
                assert page.locator('circle[data-boundary-alternative="true"]').count()>=1
                assert "coincident terminal alternative" in page.locator("#event-rows").inner_text()
                assert "signed direction +1" in page.locator("#event-rows").inner_text()
                assert "post-stop motion" in page.locator("body").inner_text()
                assert all(x["status"]=="unpaired" for item in evidence["entries"]
                           for x in item["event_time_comparisons"])
            if case=="dense-terminal":
                assert len(run.plan.questions)==16
                native=[run.baseline_run]+[item.run for item in run.entries]
                assert all(x.trajectory.termination=="terminal-event" for x in native)
                assert [len(x.events) for x in native]==[16,16,16]
                assert actual_roots==48
                assert page.locator("#event-timeline circle").count()==48
                assert page.locator("#event-rows tr").count()==48
                assert page.locator('circle[data-boundary-alternative="true"]').count()>=20
                assert all(item["status"]=="unpaired"
                           for e in evidence["entries"]
                           for item in e["event_time_comparisons"])
                assert "decorative" in page.locator("body").inner_text().lower()
                # Check actual SVG geometry and native stop boundary in each
                # physical scenario; counting DOM nodes alone missed F-05.
                geometry=page.evaluate("""() => {
                    const svg=document.getElementById('event-timeline');
                    const width=svg.viewBox.baseVal.width;
                    const height=svg.viewBox.baseVal.height;
                    return [...svg.querySelectorAll('circle')].map(el=>({
                        id:el.dataset.eventId,
                        lane:Number(el.dataset.laneIndex),
                        laneY:Number(el.dataset.laneCentreY),
                        stopX:Number(el.dataset.stopBoundaryX),
                        x:Number(el.getAttribute('cx')),
                        y:Number(el.getAttribute('cy')),
                        r:Number(el.getAttribute('r')),
                        alternative:el.dataset.boundaryAlternative==='true',
                        width,height
                    }));
                }""")
                assert len(geometry)==48
                for index,track in enumerate(native):
                    laneY=35+index*46
                    expectedX=190+track.trajectory.samples[-1].time_s*722/(
                        run.plan.sweep.baseline.spec.duration_s)
                    marks=[p for p in geometry if p["lane"]==index]
                    assert len(marks)==16,(index,len(marks))
                    assert {p["id"] for p in marks}=={
                        f"cluster:{n:02}" for n in range(16)}
                    for marker in marks:
                        assert marker["laneY"]==laneY
                        assert abs(marker["stopX"]-expectedX)<1e-5
                        assert 0<=marker["x"]-marker["r"]
                        assert marker["x"]+marker["r"]<=marker["width"]
                        assert 0<=marker["y"]-marker["r"]
                        assert marker["y"]+marker["r"]<=marker["height"]
                        assert abs(marker["y"]-laneY)+marker["r"]<23, marker
                        if marker["alternative"]:
                            assert expectedX-38<=marker["x"]<=expectedX
            if case=="gravity-guard":
                assert evidence["baseline_termination"]=="source-guard"
                assert evidence["entries"][0]["event_run"]["trajectory"]["termination"]=="duration"
                assert evidence["baseline_outcomes"]["far:detector"]=="search-truncated"
                assert "not-found-at-requested-resolution" in page.locator("#event-rows").inner_text()
            page.screenshot(path=str(output/(case+"-wide.png")),full_page=True)
            page.set_viewport_size({"width":390,"height":844})
            page.screenshot(path=str(output/(case+"-mobile.png")),full_page=True)
            mobile=page.evaluate("""() => ({
                viewport:window.innerWidth,
                document:document.documentElement.scrollWidth,
                timeline:document.getElementById('event-timeline').getBoundingClientRect().width
            })""")
            assert mobile["viewport"]==390 and mobile["document"]<=390,mobile
            assert mobile["timeline"]<=390,mobile
            cases[case]=dict(status="pass",source_ids=[
                source_id for source_id,label in run.plan.sweep.baseline.source_annotations],
                questions=[q.event_id for q in run.plan.questions],
                baseline_termination=run.baseline_run.trajectory.termination,
                final_child_terminations=[None if e.run is None else
                    e.run.trajectory.termination for e in run.entries],
                root_count=actual_roots,mobile=mobile)
            page.close()
        browser.close()
    assert not failures and not consoles and not requests,(failures,consoles,requests)
    result=dict(status="pass",cases=cases,page_errors=failures,
                console_errors=consoles,network_requests=requests,
                provenance="Each model is a separate teacher physical hypothetical; "
                           "native Python M24a localized the events, not JavaScript.")
    (output/"m28b-browser.json").write_text(
        json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/m28b-browser"))
    args=parser.parse_args()
    print(json.dumps(check(args.output),sort_keys=True))


if __name__=="__main__":
    main()
