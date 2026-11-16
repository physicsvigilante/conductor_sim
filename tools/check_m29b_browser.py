"""M29b real offline Chromium: truthful structural event questions and source identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from conductor_sim import UniformElectricField
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.structural_family import StructuralCase, StructuralFamilyPlan
from conductor_sim.structural_event_family import (
    StructuralEventFamilyPlan, run_structural_event_family,
)
from conductor_sim.structural_event_viewer import write_structural_event_family_html
from examples.m29b_structural_events import CASES, make_plan, run_case


def hostile_native_case():
    """Real physical source edits carrying literal (not executable) source labels."""
    plan=make_plan("uniform-detector")
    base=plan.family.baseline
    original=PrescribedScenario(
        "m29b:hostile-browser",base.spec,
        (("field:E0","<img src=x onerror=alert(1)>"),
         ("field:Ey","Named original transverse electric field")),
        'Teacher <img src=x onerror=alert(1)> </script><script>window.pwned=true</script>')
    case=StructuralCase(
        "new-source","Teacher adds another held electric field.",
        add_sources=(NamedSource(
            "field:extra","<svg onload=window.pwned=true>",
            UniformElectricField((100.,0.,0.))),))
    return run_structural_event_family(StructuralEventFamilyPlan(
        StructuralFamilyPlan("m29b:hostile-browser-plan",original,(case,)),
        plan.questions,plan.policy))


def _scroll_check(page):
    metrics=page.evaluate("""() => {
        const region=document.querySelector('.scroll');
        const table=region.querySelector('table');
        return {
            viewport:window.innerWidth,
            document:document.documentElement.scrollWidth,
            regionWidth:region.clientWidth,
            contentWidth:region.scrollWidth,
            tableWidth:table.getBoundingClientRect().width,
            columns:[...table.querySelectorAll('thead th')].map(
                el=>el.getBoundingClientRect().width),
            hint:getComputedStyle(document.getElementById('table-scroll-hint')).display,
            sticky:getComputedStyle(table.querySelector('tbody td')).position
        };
    }""")
    assert metrics["viewport"]==390 and metrics["document"]<=390,metrics
    assert metrics["contentWidth"]>metrics["regionWidth"]+450,metrics
    assert metrics["tableWidth"]>=1080,metrics
    assert all(width>=limit for width,limit in zip(
        metrics["columns"],(155,130,165,280,190,130))),metrics
    assert metrics["hint"]!="none" and metrics["sticky"]=="sticky",metrics
    page.locator(".scroll").evaluate("(el)=>el.scrollLeft=el.scrollWidth")
    scroll=page.locator(".scroll").evaluate("(el)=>el.scrollLeft")
    assert scroll>400,(metrics,scroll)
    assert page.evaluate("document.documentElement.scrollWidth")<=390
    return metrics


def check(output: Path):
    output.mkdir(parents=True,exist_ok=True)
    errors,consoles,requests=[],[],[]
    observed={}
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        for kind in (*CASES,"hostile-labels"):
            run=hostile_native_case() if kind=="hostile-labels" else run_case(kind)
            path=write_structural_event_family_html(
                run,output/(kind+".html"),
                title="M29b · Actual structural source event observations · "+kind)
            (output/(kind+".json")).write_text(
                json.dumps(run.as_dict(),allow_nan=False,indent=2)+"\n",
                encoding="utf-8")
            page=browser.new_page(viewport={"width":1360,"height":900})
            page.on("pageerror",lambda error:errors.append(str(error)))
            page.on("console",lambda msg:consoles.append(msg.text) if msg.type=="error" else None)
            page.on("request",lambda req:requests.append(req.url))
            page.set_content(path.read_text(encoding="utf-8"),wait_until="load")
            data=page.evaluate("""() => JSON.parse(
                document.getElementById('m29b-native-evidence').textContent)""")
            assert data["schema"]=="conductor_sim.structural_event_family.v1"
            assert data["baseline_revision_id"]==run.plan.family.baseline.revision_id
            assert data["baseline_scenario"]==run.plan.family.baseline.as_dict()
            assert data["time_interpolation"] is False
            assert data["postguard_extrapolation"] is False
            assert [e["status"] for e in data["entries"]]==[e.status for e in run.entries]
            assert len(data["questions"])==len(run.plan.questions)
            assert page.locator("#structural-event-timeline").count()==1
            assert page.locator("#source-memberships article").count()==len(run.entries)+1
            assert "EVENT-TIME" in page.locator("body").inner_text()
            assert "NOT rigorous" in json.dumps(data["qualification"]) or "NOT" in data["qualification"]
            assert "Workbook" not in page.locator("body").inner_text()
            assert "source" in page.locator("body").inner_text().lower()

            native=[run.baseline_run,*[entry.event_run for entry in run.entries]]
            scenarios=[run.plan.family.baseline,*[
                entry.assembly.scenario if entry.assembly is not None else None
                for entry in run.entries]]
            expected_roots=sum(len(n.events) for n in native if n is not None)
            expected_rows=0
            for i,(scenario,study) in enumerate(zip(scenarios,native)):
                card=page.locator("#source-memberships article").nth(i)
                text=card.inner_text()
                if scenario is None:
                    assert "No qualified physical source inventory" in text
                else:
                    for (identity,label),source in zip(
                            scenario.source_annotations,scenario.spec.field.sources):
                        assert identity in text and label in text
                        assert type(source).__name__ in text
                    if scenario.spec.gravity is not None:
                        assert "Uniform gravity present" in text
                if study is None:
                    assert "not-executed" in page.locator(
                        f'#event-rows tr[data-scenario-index="{i}"]').first.inner_text()
                else:
                    for q in run.plan.questions:
                        roots=[event for event in study.events if event.event_id==q.event_id]
                        expected_rows+=max(1,len(roots))
                        outcome=study.as_dict()["outcomes"][q.event_id]
                        affected=page.locator(f'#event-rows tr[data-scenario-index="{i}"]'
                                              f'[data-event-id="{q.event_id}"]')
                        assert affected.count()==max(1,len(roots))
                        assert outcome in affected.first.inner_text()
                        for root_index,event in enumerate(roots):
                            # The same physical detector can cross more than
                            # once. Verify EACH native root's own displayed row:
                            # Playwright strict mode correctly forbids treating
                            # multiple matches as one arbitrary first element.
                            visible=affected.nth(root_index).inner_text()
                            assert f"signed direction {event.direction:+d}" in visible
                            assert "bracket=" in visible
                if study is None:
                    expected_rows+=len(run.plan.questions)
            assert page.locator("#event-rows tr").count()==expected_rows
            assert len(data["baseline_event_run"]["events"])+sum(
                len(e["event_run"]["events"]) for e in data["entries"]
                if e["event_run"] is not None)==expected_roots

            plot=page.evaluate("""() => {
                const svg=document.getElementById('structural-event-timeline');
                const w=svg.viewBox.baseVal.width,h=svg.viewBox.baseVal.height;
                const badges=[...svg.querySelectorAll('[data-root-count]')].map(el=>({
                    lane:Number(el.dataset.laneIndex),
                    count:Number(el.dataset.rootCount),
                    stop:Number(el.dataset.stopX),
                    x:Number(el.getAttribute(el.tagName.toLowerCase()==='circle'?'cx':'x')),
                    y:Number(el.getAttribute(el.tagName.toLowerCase()==='circle'?'cy':'y')),
                    shape:el.tagName.toLowerCase()
                }));
                const lines=[...svg.querySelectorAll('[data-observed-lane]')].map(el=>({
                    lane:Number(el.dataset.observedLane),
                    end:Number(el.getAttribute('x2')),
                    y:Number(el.getAttribute('y1'))
                }));
                return {width:w,height:h,badges,lines};
            }""")
            assert sum(b["count"] for b in plot["badges"])==expected_roots
            assert len(plot["lines"])==sum(n is not None for n in native)
            for i,study in enumerate(native):
                marks=[m for m in plot["badges"] if m["lane"]==i]
                assert sum(m["count"] for m in marks)==(
                    0 if study is None else len(study.events))
                if study is None:
                    continue
                y=38+i*52
                end=187+study.trajectory.samples[-1].time_s*(
                    930-187-37)/run.plan.family.baseline.spec.duration_s
                line=next(x for x in plot["lines"] if x["lane"]==i)
                assert abs(line["end"]-end)<1e-5,(kind,i,line,end)
                assert line["y"]==y
                for mark in marks:
                    assert abs(mark["stop"]-end)<1e-5
                    assert 0<=mark["x"]<=plot["width"]
                    assert abs(mark["y"]-y)<26
            if kind=="uniform-detector":
                assert all(e.status=="complete" for e in run.entries)
                assert all(e["event_time_comparisons"][0]["status"]=="one-resolved-pair"
                           for e in data["entries"])
                assert data["entries"][0]["event_time_comparisons"][0][
                    "estimated_numerical_shift_s"]<0
            elif kind=="guard-detector":
                assert [e.status for e in run.entries]==[
                    "censored","complete","rejected-structural-model"]
                assert data["entries"][0]["event_outcomes"]["x:far"]=="search-truncated"
                assert data["entries"][0]["event_time_comparisons"][0]["status"]=="unpaired"
                assert "censored" in data["entries"][0]["event_time_comparisons"][0]["reason"]
            elif kind=="terminal-refusal":
                assert [e.status for e in run.entries]==[
                    "rejected-event-study","complete"]
                assert data["entries"][0]["structural_assembly"] is not None
                assert data["entries"][0]["event_run"] is None
            elif kind=="double-crossing":
                assert data["entries"][0]["event_time_comparisons"][0]["status"]=="unpaired"
                assert expected_roots==4
            elif kind=="coincident-terminal":
                assert expected_roots==48
                assert len(plot["badges"])<=6
                assert all(e.status=="censored" for e in run.entries)
                assert all(p["status"]=="unpaired" for e in data["entries"]
                           for p in e["event_time_comparisons"])
                assert page.locator("#event-rows tr").count()==48
            elif kind=="mirror-detector":
                assert data["entries"][1]["status"]=="rejected-structural-model"
            elif kind=="hostile-labels":
                visible=page.locator("#source-memberships").inner_text()
                assert "<img src=x onerror=alert(1)>" in visible
                assert "<svg onload=window.pwned=true>" in visible
                assert page.locator("img").count()==0
                assert page.locator("svg").count()==1
                assert page.locator("script").count()==1
                assert page.evaluate("typeof window.pwned")=="undefined"
            page.screenshot(path=str(output/(kind+"-wide.png")),full_page=True)
            page.set_viewport_size({"width":390,"height":844})
            readability=_scroll_check(page)
            page.screenshot(path=str(output/(kind+"-mobile.png")),full_page=True)
            observed[kind]=dict(status="pass",root_count=expected_roots,
                table_rows=expected_rows,
                source_ids=[sid for sid,label in run.plan.family.baseline.source_annotations],
                case_statuses=[e.status for e in run.entries],
                native_terminations=[
                    None if n is None else n.trajectory.termination for n in native],
                plot_badges=len(plot["badges"]),mobile=readability)
            page.close()
        browser.close()
    assert not errors and not consoles and not requests,(errors,consoles,requests)
    report=dict(status="pass",cases=observed,
                page_errors=errors,console_errors=consoles,
                network_requests=requests,
                provenance=("Real independently reassembled native source inventories "
                            "and M24a event roots; grouped SVG markers are decorative, "
                            "no browser integration or extrapolation."))
    (output/"m29b-browser.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/m29b-browser"))
    args=parser.parse_args()
    print(json.dumps(check(args.output),sort_keys=True))


if __name__=="__main__":
    main()
