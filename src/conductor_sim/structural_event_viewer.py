"""M29b source-agnostic offline event-time chart over source membership revisions.

All root times, outcomes, original brackets and source inventories come from
already executed native M24a studies. Grouped SVG glyphs are decorative
counts, NEVER an ODE approximation or replacement for the native root table.
"""
from __future__ import annotations

from html import escape
import json
from math import isfinite
from pathlib import Path

from .motion_events import validate_event_run
from .structural_event_family import StructuralEventFamilyRun


def _h(value):
    return escape(str(value), quote=True)


def _fmt(value):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("nonfinite physical event/chart data")
    return f"{value:.16g}"


def _rows(run):
    original=run.plan.family.baseline
    rows=[dict(
        scenario=original, id=original.scenario_id,
        status=("complete" if run.baseline_run.trajectory.termination=="duration"
                else "censored"), reason=run.baseline_run.trajectory.termination,
        event_run=run.baseline_run, structural_changes=(),
        original=True, comparisons={})]
    for entry in run.entries:
        assembly=entry.assembly
        record=entry.as_dict(run.baseline_run,run.plan.questions)
        rows.append(dict(
            scenario=None if assembly is None else assembly.scenario,
            id=(f"{run.plan.family.plan_id}:{entry.case.case_id}"
                if assembly is None else assembly.scenario.scenario_id),
            status=entry.status, reason=entry.reason,
            event_run=entry.event_run,
            original=False,
            structural_changes=() if assembly is None else assembly.changes,
            comparisons={c["event_id"]:c for c in record["event_time_comparisons"]}))
    return rows


def _chart(rows, horizon):
    """Group close native roots in each own lane; preserve original numeric data in table."""
    width,left,right=930,187,37
    n=len(rows)
    height=88+n*52
    scale=(width-left-right)/horizon
    colors=("#126994","#b65b35","#567e42","#845ba6","#9b742a","#16827a")
    out=[f'<svg id="structural-event-timeline" viewBox="0 0 {width} {height}" '
         'role="img" aria-label="Native event observation horizons; grouped '
         'event-count markers are NOT exact event times">'
         '<title>Actual observed physical times and grouped numerical roots</title>'
         '<desc>Each colored line ends at its last original native physical time. '
         'Markers of nearby original numerical root times are grouped by screen '
         'distance; their positions are decorative, not physical interpolation. '
         'All original event brackets and times remain in the table.</desc>']
    for j in range(5):
        t=horizon*j/4
        x=left+(width-left-right)*j/4
        out.append(f'<line x1="{x:.7f}" x2="{x:.7f}" y1="16" '
                   f'y2="{height-30}" stroke="#e1e8ef" stroke-width="1"/>')
        out.append(f'<text x="{x:.7f}" y="{height-9}" fill="#496274" '
                   f'text-anchor="middle" font-size="12">{_h(_fmt(t))} s</text>')
    for i,row in enumerate(rows):
        y=38+i*52
        color=colors[i]
        out.append(f'<text x="7" y="{y+4}" fill="#223d52" font-size="11.5">'
                   f'{_h(row["id"][:27])}</text>')
        native=row["event_run"]
        if native is None:
            out.append(f'<text x="{left+4}" y="{y+4}" font-size="12" '
                       f'fill="#925050">{_h(row["status"])}</text>')
            continue
        validate_event_run(native)
        final=native.trajectory.samples[-1].time_s
        if not 0<=final<=horizon:
            raise ValueError("native event run has invalid observed time horizon")
        stop_x=left+final*scale
        out.append(f'<line data-observed-lane="{i}" x1="{left}" '
                   f'x2="{stop_x:.7f}" y1="{y}" y2="{y}" '
                   f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        # All roots are preserved in M24a's actual event list; only near
        # coincident *screen glyphs* are collapsed to count badges.
        # M24a sometimes retains overlapping terminal candidates whose
        # representative numerical time is just after the real selected stop.
        # Their drawn position is always the genuine stop BOUNDARY, never
        # later physical motion.
        root_glyphs=[]
        for event in native.events:
            at=event.sample.time_s
            if not 0<=at<=horizon:
                raise ValueError("native event root outside declared duration")
            draw_time=min(at,final)
            x=left+draw_time*scale
            root_glyphs.append((x,event,at>final))
        root_glyphs.sort(key=lambda item:(item[0],item[1].event_id))
        groups=[]
        for item in root_glyphs:
            if groups and item[0]-groups[-1][-1][0] <= 15:
                groups[-1].append(item)
            else:
                groups.append([item])
        glyphs=0
        for group in groups:
            x=group[0][0]  # first native visible root or selected stop boundary
            count=len(group)
            later=sum(bool(e[2]) for e in group)
            label=(f"{count} original numerical event root(s) in this source case; "
                   f"{later} terminal-boundary alternative(s) not physical poststop "
                   "motion; grouped glyph position is not an event sample time. "
                   "Read exact native brackets/directions in table. IDs: "+
                   ", ".join(z[1].event_id for z in group))
            if count==1:
                out.append(f'<circle data-lane-index="{i}" '
                           f'data-root-count="1" data-stop-x="{stop_x:.7f}" '
                           f'cx="{x:.7f}" cy="{y}" r="5" fill="{color}" '
                           'stroke="#fff" stroke-width="1.3">'
                           f'<title>{_h(label)}</title></circle>')
            else:
                # One bounded per-lane count badge instead of unbounded
                # vertical alternatives (the old M28b F-05 defect).
                bx=max(left+2,min(x-12,width-right-28))
                out.append(f'<rect data-lane-index="{i}" data-root-count="{count}" '
                           f'data-stop-x="{stop_x:.7f}" x="{bx:.7f}" y="{y-9}" '
                           f'width="28" height="18" rx="5" fill="{color}">'
                           f'<title>{_h(label)}</title></rect>')
                out.append(f'<text x="{bx+14:.7f}" y="{y+4.5}" '
                           f'fill="#fff" text-anchor="middle" pointer-events="none" '
                           f'font-size="11" font-weight="600">{count}</text>')
            glyphs+=count
        if glyphs!=len(native.events):
            raise ValueError("visual grouped markers lost original numerical roots")
        if native.trajectory.termination!="duration":
            out.append(f'<rect data-terminal-lane="{i}" '
                       f'x="{stop_x-4:.7f}" y="{y+13}" width="8" height="8" '
                       'fill="#b83935"><title>'
                       f'{_h(native.trajectory.termination)} at {_h(_fmt(final))} s'
                       '</title></rect>')
    out.append("</svg>")
    return "".join(out)


def render_structural_event_family(
        run: StructuralEventFamilyRun, *,
        title="M29b · Physical source arrangements and event outcomes") -> str:
    if type(run) is not StructuralEventFamilyRun:
        raise TypeError("requires native StructuralEventFamilyRun, not untrusted JSON")
    if (type(title) is not str or not 1<=len(title)<=120 or
            title!=title.strip() or not title.isprintable()):
        raise ValueError("event family report title must be printable and trimmed")
    data=run.as_dict()
    original=run.plan.family.baseline
    rows=_rows(run)
    horizon=original.spec.duration_s
    chart=_chart(rows,horizon)
    cards=[]
    event_tables=[]
    for i,row in enumerate(rows):
        scenario=row["scenario"]
        native=row["event_run"]
        if scenario is None:
            source_list="<li>No qualified physical source inventory: refused</li>"
            provenance="Teacher hypothetical requested; whole-model qualification refused"
            membership="No qualified membership or gravity state"
        else:
            entries=(
                f'<li>{_h(sid)} — {_h(label)} ({_h(type(source).__name__)})</li>'
                for (sid,label),source in zip(
                    scenario.source_annotations,scenario.spec.field.sources))
            source_list="".join(entries) or "<li>No held electric source components</li>"
            provenance=f"{scenario.origin} — {scenario.provenance}"
            membership=("Uniform gravity present" if scenario.spec.gravity is not None
                        else "No gravity component")
        changes=[]
        for change in row["structural_changes"]:
            before=change.get("before")
            after=change.get("after")
            before_id=(None if before is None else
                       before.get("id",before.get("component_id")))
            after_id=(None if after is None else
                      after.get("id",after.get("component_id")))
            changes.append(f"{change['operation']}: {before_id or 'none'} → {after_id or 'none'}")
        delta="; ".join(changes) if changes else "Original unchanged apparatus" if i==0 else "Structural change refused"
        cards.append(
            f'<article class="inventory" data-scenario-index="{i}">'
            f'<h3>{_h(row["id"])}</h3>'
            f'<p class="tiny">{"Original model" if i==0 else "Independent teacher hypothetical"} '
            f'· <strong>{_h(row["status"])}</strong> · {_h(row["reason"])}</p>'
            f'<p class="tiny">Authored provenance assertion: {_h(provenance)}</p>'
            f'<ul>{source_list}</ul>'
            f'<p class="tiny">{_h(membership)}. Real structural ledger: {_h(delta)}</p>'
            '</article>')
        for q in run.plan.questions:
            qid=q.event_id
            if native is None:
                outcome="not-executed"
                actual=[]
                stop=row["status"]
            else:
                outcome=native.as_dict()["outcomes"][qid]
                actual=[e for e in native.events if e.event_id==qid]
                end=native.trajectory.samples[-1].time_s
                stop=f"{native.trajectory.termination} at {_fmt(end)} s"
            comparison=row["comparisons"].get(qid)
            if i==0: paired="Original reference event"
            elif comparison is None: paired="No native event-study comparison"
            elif comparison["status"]=="one-resolved-pair":
                lo,hi=comparison["numerical_bracket_shift_interval_s"]
                paired=(f"Estimated numerical Δt={_fmt(comparison['estimated_numerical_shift_s'])} s; "
                        f"bracket difference [{_fmt(lo)}, {_fmt(hi)}] s")
            else: paired="Unpaired: "+comparison["reason"]
            root_list=actual if actual else [None]
            for event in root_list:
                if event is None:
                    reading=("No located root; not a proof of eternal absence"
                             if outcome=="not-found-at-requested-resolution"
                             else "No located root in observed horizon")
                else:
                    lo,hi=event.time_bracket_s
                    original_time=event.sample.time_s
                    note=("Overlapping terminal alternative: original numerical candidate "
                          "is NOT after-stop physical motion"
                          if original_time>native.trajectory.samples[-1].time_s
                          else "Located numerical event")
                    reading=(f"t≈{_fmt(original_time)} s; "
                             f"bracket=[{_fmt(lo)}, {_fmt(hi)}] s; "
                             f"signed direction {event.direction:+d}; "
                             f"residual {_fmt(event.residual)} {event.residual_unit}; {note}")
                event_tables.append(
                    f'<tr data-scenario-index="{i}" data-event-id="{_h(qid)}">'
                    f'<td>{_h(row["id"])}</td><td>{_h(qid)}</td>'
                    f'<td>{_h(outcome)}</td><td>{_h(reading)}</td>'
                    f'<td>{_h(paired)}</td><td>{_h(stop)}</td></tr>')
    safe=(json.dumps(data,allow_nan=False,separators=(",",":"))
          .replace("&",r"\u0026").replace("<",r"\u003c")
          .replace(">",r"\u003e"))
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>"""+_h(title)+"""</title>
<style>
:root{font:15px/1.5 system-ui,sans-serif;color:#183147;background:#f4f7fa}
*{box-sizing:border-box}body{margin:0}
header{padding:20px max(16px,calc((100vw - 1120px)/2));background:#173c54;color:white}
h1{font-size:23px;margin:0 0 7px}header p{margin:0;font-size:12px;color:#dceaf2}
main{max-width:1150px;margin:auto;padding:14px;display:grid;gap:15px}
section{background:#fff;border:1px solid #cfdae4;border-radius:8px;
padding:16px;min-width:0;overflow-wrap:anywhere}
h2{font-size:17px;margin:0 0 8px}h3{font-size:14px;margin:0 0 6px;overflow-wrap:anywhere}
p{margin:5px 0 9px}.tiny{font-size:12px;color:#546a7d;overflow-wrap:anywhere}
svg{width:100%;height:auto;display:block;border:1px solid #e0e7ee;border-radius:5px;background:#fbfdff}
.inventory-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr));gap:10px}
.inventory{border:1px solid #dce5ec;border-radius:6px;padding:12px;min-width:0}
.inventory ul{padding-left:20px;margin:8px 0;font-size:12px}
.inventory li{overflow-wrap:anywhere;margin-bottom:3px}
.scroll{max-width:100%;overflow-x:auto;overscroll-behavior-inline:contain}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #e1e8ee;overflow-wrap:anywhere}
th{background:#edf3f7}
.scroll-hint{display:none}
@media(max-width:520px){
  main{padding:10px}section{padding:11px}h1{font-size:19px}
  .scroll-hint{display:block}
  .scroll table{table-layout:fixed;min-width:1080px}
  .scroll th:nth-child(1){width:160px}
  .scroll th:nth-child(2){width:135px}
  .scroll th:nth-child(3){width:170px}
  .scroll th:nth-child(4){width:285px}
  .scroll th:nth-child(5){width:195px}
  .scroll th:nth-child(6){width:135px}
  .scroll th:first-child,.scroll td:first-child{
    position:sticky;left:0;z-index:1;background:#fff;
    box-shadow:2px 0 3px rgba(24,62,88,.15)
  }
  .scroll th:first-child{z-index:2;background:#edf3f7}
}
</style></head><body><header><h1>"""+_h(title)+"""</h1>
<p>Separate physical source inventories · true native typed event roots · no browser ODE</p>
</header><main>
<section><h2>Numerical event observation horizons</h2>
<p class="tiny">Each line ends at its actual native solved physical time.
Markers at very close times are combined into a compact event COUNT, not
an interpolated physical event time; the original individual event times,
signed directions and numerical brackets remain in the full table below.
A red square marks a terminal/guard/unresolved observation. No field source
moves during a run, and there is NO physical motion after a stop.</p>"""+chart+"""</section>
<section><h2>Actual authored source inventories for each experiment</h2>
<p class="tiny">These are separate hypothetical experiments, not changes
to the apparatus during one particle trajectory. Source ID/name/type and
M22a structural lineage refer to real native qualified models. The source
provenance is an authored assertion, not independently authenticated PDF data.</p>
<div class="inventory-grid" id="source-memberships">"""+"".join(cards)+"""</div></section>
<section><h2>Every event question, native root and time comparison</h2>
<p class="tiny">A numerical root-time shift is eligible ONLY when exactly one
resolved root with the same signed direction exists in both duration-complete
histories. Two crossings, opposite direction, guard, terminal stop, or unresolved
queries remain unpaired. Root brackets are numerical localization, NOT exact-ODE
uncertainty bounds. No cross-inventory potential energy is interpreted as work.</p>
<p class="tiny scroll-hint" id="table-scroll-hint">Scroll the table horizontally
to read event brackets, complete reason and source-specific outcome;
the physical scenario column stays visible.</p>
<div class="scroll" role="region" tabindex="0" aria-label="Structural event-family numerical roots and scenario comparisons; scroll horizontally">
<table><thead><tr><th>Scenario</th><th>Question ID</th><th>Outcome</th>
<th>Original numerical root / bracket / direction</th><th>Comparison to baseline</th>
<th>Observation ending</th></tr></thead><tbody id="event-rows">"""+"".join(event_tables)+"""</tbody></table></div>
</section>
<section><h2>Scientific boundaries</h2>
<p>These model changes add, remove or replace fixed prescribed physical
sources or gravity in separately qualified scenarios. The diagram is an
EVENT-TIME summary, not the source apparatus, particle path, a conducting
metal model, induced charge, source contact, or a time-dependent field.
There is no browser physics, no interpolation, no unqualified 3D particle
motion and no implied collision law. Source labels are author-provided
assertions; the numerical root evidence remains available verbatim in the
detached native JSON report.</p>
</section></main>
<script type="application/json" id="m29b-native-evidence">"""+safe+"""</script>
</body></html>"""


def write_structural_event_family_html(
        run: StructuralEventFamilyRun, path, **kwargs) -> Path:
    output=Path(path)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(render_structural_event_family(run,**kwargs),
                      encoding="utf-8")
    return output
