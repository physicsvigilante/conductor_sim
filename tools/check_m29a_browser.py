"""M29a real Chromium acceptance: correct native trajectories and source membership."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from examples.m29a_structural_family import CASES, run_case, model
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.structural_family import (
    StructuralCase, StructuralFamilyPlan, run_structural_family,
)
from conductor_sim.prescribed_fields import UniformElectricField
from conductor_sim.structural_family_viewer import write_structural_family_html


def hostile_source_labels_case():
    """A qualified model containing two intentionally distinct literal hostile source names."""
    base=model("uniform-edits").baseline
    original=PrescribedScenario(
        "m29a:hostile-labels",base.spec,
        (("field:E0","<img src=x onerror=alert(1)>"),
         ("field:Ey","Named held transverse electric field")),
        'Teacher <img src=x onerror=alert(1)> </script><script>window.pwned=true</script>')
    case=StructuralCase(
        "added-source","Teacher adds an independently named physical source.",
        add_sources=(NamedSource(
            "field:Enew","<svg onload=window.pwned=true>",
            UniformElectricField((6.,0.,0.))),))
    return run_structural_family(StructuralFamilyPlan(
        "m29a:hostile-browser",original,(case,)))


def mobile_readability(page):
    """Read actual table geometry and prove keyboard-scannable local scrolling."""
    metrics=page.evaluate("""() => {
        const region=document.querySelector('.scroll');
        const table=region.querySelector('table');
        return {
          viewport:window.innerWidth,document:document.documentElement.scrollWidth,
          client:region.clientWidth,scroll:region.scrollWidth,
          table:table.getBoundingClientRect().width,
          columns:[...table.querySelectorAll('thead th')].map(
            x=>x.getBoundingClientRect().width),
          sticky:getComputedStyle(table.querySelector('tbody td')).position,
          hint:getComputedStyle(document.getElementById('table-scroll-hint')).display
        };
    }""")
    assert metrics["viewport"]==390 and metrics["document"]<=390,metrics
    assert metrics["scroll"]>metrics["client"]+450,metrics
    assert metrics["table"]>=1060,metrics
    assert all(width>=minimum for width,minimum in zip(
        metrics["columns"],(165,215,180,225,230))),metrics
    assert metrics["sticky"]=="sticky" and metrics["hint"]!="none",metrics
    page.locator(".scroll").evaluate("(element)=>element.scrollLeft=element.scrollWidth")
    position=page.locator(".scroll").evaluate("(element)=>element.scrollLeft")
    assert position>400,(metrics,position)
    assert page.evaluate("document.documentElement.scrollWidth")<=390
    page.locator(".scroll").evaluate("(element)=>element.scrollLeft=0")
    return metrics


def check(output: Path):
    output.mkdir(parents=True,exist_ok=True)
    errors,consoles,requests=[],[],[]
    results={}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        for case in CASES:
            native=run_case(case)
            path=write_structural_family_html(
                native,output/(case+".html"),
                title=f"M29a · Independently assembled physical sources · {case}")
            (output/(case+".json")).write_text(
                json.dumps(native.as_dict(),indent=2,allow_nan=False)+"\n",
                encoding="utf-8")
            page=browser.new_page(viewport={"width":1360,"height":900})
            page.on("pageerror",lambda e:errors.append(str(e)))
            page.on("console",lambda e:consoles.append(e.text) if e.type=="error" else None)
            page.on("request",lambda r:requests.append(r.url))
            page.set_content(path.read_text(encoding="utf-8"),wait_until="load")
            page.wait_for_function("window.m29aStructuralViewer && window.m29aStructuralViewer.loaded")
            data=page.evaluate("""() => JSON.parse(document.getElementById(
                'm29a-native-data').textContent)""")
            assert data["schema"]=="conductor_sim.structural_family_view.v1"
            assert data["browser_ode"] is False
            assert data["time_interpolation"] is False
            assert data["extrapolation"] is False
            assert data["source_geometry_rendered"] is False
            assert data["baseline_revision_id"]==native.plan.baseline.revision_id
            assert data["baseline_provenance"]==native.plan.baseline.provenance
            assert "author assertion" in page.locator("body").inner_text()
            assert "TRAJECTORY SPACE ONLY" in page.locator("body").inner_text()
            assert "Workbook" not in page.locator("body").inner_text()
            assert page.locator("#structural-states tr").count()==len(native.entries)+1
            assert len(data["tracks"])==len(native.entries)+1
            assert data["frame_times_s"]==sorted({
                row["time_s"] for track in data["tracks"] for row in track["samples"]})
            assert page.evaluate("window.m29aStructuralViewer.physicalTime")==0
            assert data["tracks"][0]["source_inventory"]==[
                dict(id=identity,label=label,kind=type(src).__name__)
                for (identity,label),src in zip(
                    native.plan.baseline.source_annotations,
                    native.plan.baseline.spec.field.sources)]
            # The real UI, not only embedded JSON, must display all three
            # authored source attributes (physical ID, name and source type).
            for i,track in enumerate(data["tracks"]):
                inventory=track["source_inventory"]
                visible=page.locator("#structural-states tr").nth(i).locator("td").nth(1).inner_text()
                if inventory is None:
                    assert "No qualified source inventory" in visible
                else:
                    for member in inventory:
                        assert member["id"] in visible
                        assert member["kind"] in visible
                        assert member["label"] in visible
            native_saved=sum(bool(t["samples"]) for t in data["tracks"])
            assert page.locator("#structural-plot polyline").count()==native_saved
            assert page.locator('#structural-plot circle[data-marker-index]').count()==native_saved
            if case=="uniform-edits":
                assert [e.status for e in native.entries]==["complete"]*4
                assert data["source_counts"]==[2,3,1,2,2]
                assert data["tracks"][1]["source_inventory"][2]["id"]=="field:Enew"
                assert data["tracks"][2]["source_inventory"]==[
                    data["tracks"][0]["source_inventory"][0]]
                assert data["tracks"][3]["has_gravity"] is True
                assert [s["id"] for s in data["tracks"][4]["source_inventory"]]==[
                    "field:E0","field:Ey-new"]
            elif case=="point-replace":
                assert [e.status for e in native.entries]==["complete","complete"]
                assert data["tracks"][1]["source_inventory"][0]["id"]=="sphere:S"
                assert data["tracks"][1]["source_inventory"][0]["kind"]=="UniformSphereField"
                assert data["tracks"][2]["source_inventory"]==[]
                assert "replace-source" in page.locator("#structural-states").inner_text()
            elif case=="mirror-pair":
                assert [e.status for e in native.entries]==[
                    "complete","rejected-structural-model","complete"]
                assert data["source_counts"]==[1,3,None,2]
                assert data["tracks"][2]["source_inventory"] is None
                assert "rejected-structural-model" in page.locator(
                    "#structural-states").inner_text()
            elif case=="guard-and-refusal":
                assert [e.status for e in native.entries]==[
                    "censored","complete","rejected-structural-model"]
                assert data["tracks"][1]["termination"]=="source-guard"
                assert data["tracks"][2]["termination"]=="duration"
                assert data["tracks"][3]["source_inventory"] is None
            page.locator("#play").click()
            page.wait_for_function("window.m29aStructuralViewer.frameIndex>0",timeout=5000)
            page.locator("#play").click()
            page.locator("#saved-time").evaluate("""el=>{
                el.value=el.max;
                el.dispatchEvent(new Event('input',{bubbles:true}));
            }""")
            end=page.evaluate("window.m29aStructuralViewer.physicalTime")
            assert end==data["frame_times_s"][-1]
            active=page.evaluate("window.m29aStructuralViewer.activeCount")
            unavailable=page.evaluate("window.m29aStructuralViewer.unavailableCount")
            assert active+unavailable==len(data["tracks"])
            if case=="guard-and-refusal":
                assert active==2 and unavailable==2
                assert "No saved state" in page.locator("#structural-states").inner_text()
                assert "No literal-time comparison" in page.locator(
                    "#structural-states").inner_text()
                assert page.locator('#structural-plot circle[data-marker-index="1"]').evaluate(
                    "(e)=>getComputedStyle(e).display")=="none"
            else:
                assert active==native_saved
                assert "Δr" in page.locator("#structural-states").inner_text()
            page.screenshot(path=str(output/(case+"-wide.png")),full_page=True)
            page.locator("#restart").click()
            assert page.evaluate("window.m29aStructuralViewer.physicalTime")==0
            page.set_viewport_size({"width":390,"height":844})
            page.screenshot(path=str(output/(case+"-mobile.png")),full_page=True)
            mobile=page.evaluate("""() => ({
                viewport:window.innerWidth,
                document:document.documentElement.scrollWidth,
                svg:document.getElementById('structural-plot').getBoundingClientRect().width
            })""")
            if mobile["document"]>390:
                bad=page.evaluate("""() => [...document.querySelectorAll('*')]
                  .map(e=>({tag:e.tagName,id:e.id,x:e.getBoundingClientRect().right}))
                  .filter(e=>e.x>window.innerWidth+1).slice(0,12)""")
                raise AssertionError((case,mobile,bad))
            assert mobile["viewport"]==390 and mobile["document"]<=390,mobile
            assert mobile["svg"]<=390,mobile
            readability=mobile_readability(page)
            results[case]=dict(status="pass",source_counts=data["source_counts"],
                case_statuses=[e.status for e in native.entries],
                frame_count=len(data["frame_times_s"]),last_time_s=end,
                active_at_end=active,unavailable_at_end=unavailable,
                mobile=mobile,readability=readability)
            page.close()

        # Full native source-label injection test in the actual browser:
        # original and new source IDs/types/labels all must remain VISIBLE
        # and inert, not merely serialized inside embedded JSON.
        hostile=hostile_source_labels_case()
        assert hostile.entries[0].status=="complete"
        html=write_structural_family_html(
            hostile,output/"hostile-labels.html",
            title="M29a · Source names and document-injection regression")
        page=browser.new_page(viewport={"width":390,"height":844})
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.on("console",lambda e:consoles.append(e.text) if e.type=="error" else None)
        page.on("request",lambda r:requests.append(r.url))
        page.set_content(html.read_text(encoding="utf-8"),wait_until="load")
        page.wait_for_function("window.m29aStructuralViewer && window.m29aStructuralViewer.loaded")
        data=page.evaluate("""() => JSON.parse(
            document.getElementById('m29a-native-data').textContent)""")
        assert data["tracks"][0]["source_inventory"][0]["label"]=="<img src=x onerror=alert(1)>"
        assert data["tracks"][1]["source_inventory"][-1]["label"]=="<svg onload=window.pwned=true>"
        baseline_members=page.locator("#structural-states tr").nth(0).locator("td").nth(1).inner_text()
        child_members=page.locator("#structural-states tr").nth(1).locator("td").nth(1).inner_text()
        assert "<img src=x onerror=alert(1)>" in baseline_members
        assert "Named held transverse electric field" in baseline_members
        assert "<svg onload=window.pwned=true>" in child_members
        assert "field:Enew" in child_members and "UniformElectricField" in child_members
        assert page.locator("img").count()==0
        assert page.locator("svg").count()==1
        assert page.locator("script").count()==2
        assert page.evaluate("typeof window.pwned")=="undefined"
        assert "<img" in page.locator("#provenance").inner_text()
        readability=mobile_readability(page)
        page.screenshot(path=str(output/"hostile-labels-mobile.png"),full_page=True)
        results["hostile-labels"]=dict(
            status="pass",visible_baseline_label=True,
            visible_added_label=True,escaped_literal_html=True,
            mobile_readability=readability)
        page.close()
        browser.close()
    assert not errors and not consoles and not requests,(errors,consoles,requests)
    report=dict(status="pass",cases=results,page_errors=errors,
                console_errors=consoles,network_requests=requests,
                provenance="Native independently reassembled physical sources and saved "
                           "trajectories, not browser-time equations or changing sources.")
    (output/"m29a-browser.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/m29a-browser"))
    args=parser.parse_args()
    print(json.dumps(check(args.output),sort_keys=True))


if __name__=="__main__":
    main()
