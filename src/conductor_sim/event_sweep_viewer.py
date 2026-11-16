"""M28b: source-agnostic event sweep report and true numerical event-time chart.

The HTML reports native M24a event brackets and their actual observation
horizons. It draws neither particle paths nor ring apparatus geometry, and
performs no numerical event search or trajectory integration in the browser.
"""
from __future__ import annotations

from html import escape
import json
from math import isfinite
from pathlib import Path

from .event_sweep import EventSweepRun
from .motion_events import validate_event_run


def _h(value):
    return escape(str(value), quote=True)


def _fmt(value):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("nonfinite event display value")
    return f"{value:.16g}"


def _root_scope(event, last_time):
    if event.sample.time_s > last_time:
        return "coincident terminal alternative at stop boundary, not post-stop motion"
    return "terminal observation" if event.terminal else "passive observation"


def _statuses(run: EventSweepRun):
    all_rows = [dict(
        id=run.plan.sweep.baseline.scenario_id,
        value="original",
        scenario=run.plan.sweep.baseline,
        event_run=run.baseline_run,
        status=("complete" if run.baseline_run.trajectory.termination == "duration"
                else "censored"),
        reason=run.baseline_run.trajectory.termination,
        shifts={})]
    for e in run.entries:
        data = e.as_dict(run.baseline_run, run.plan.questions)
        all_rows.append(dict(
            id=(e.scenario.scenario_id if e.scenario is not None else
                f"{run.plan.sweep.plan_id}:v{e.index:02d}"),
            value=repr(e.requested_value), scenario=e.scenario, event_run=e.run,
            status=e.status, reason=e.reason,
            shifts={x["event_id"]: x for x in data["event_time_comparisons"]}))
    return all_rows


def _timeline(rows, run):
    """Numerical time windows only; no line after a stopped/censored run."""
    horizon = run.plan.sweep.baseline.spec.duration_s
    n = len(rows)
    width, height = 960, 92 + n*46
    left, right = 190, 48
    scale = (width - left - right)/horizon
    colors = ("#136b94", "#b45c39", "#397a4e", "#8262ac",
              "#a7782e", "#257f7d", "#a3446b")
    s = [f'<svg id="event-timeline" viewBox="0 0 {width} {height}" '
         'role="img" aria-label="Numerical event occurrences and observation horizons">'
         '<title>Physical event-time intervals and observed stop times</title>',
         '<desc>Lines stop at each native run last time; circles are individual '
         'located M24a numerical event roots, not physical collisions. '
         'Numbers and root brackets are listed in the table.</desc>']
    for i in range(5):
        t = horizon*i/4
        x = left+t*scale
        s.append(f'<line x1="{x:.6f}" x2="{x:.6f}" y1="18" '
                 f'y2="{height-28}" stroke="#dbe3ed" stroke-width="1"/>')
        s.append(f'<text x="{x:.6f}" y="{height-8}" text-anchor="middle" '
                 f'fill="#53697e" font-size="12">{_h(_fmt(t))} s</text>')
    for i, entry in enumerate(rows):
        y = 35 + i*46
        name = entry["id"]
        color = colors[i]
        s.append(f'<text x="8" y="{y+4}" fill="#263e55" font-size="12">'
                 f'{_h(name[:25])}</text>')
        native = entry["event_run"]
        if native is not None:
            # M24a owns the special overlapping-terminal-alternative contract.
            # Validate it rather than treating its alternate root as motion.
            validate_event_run(native)
        if native is None:
            s.append(f'<text x="{left}" y="{y+4}" fill="#8b5757" font-size="12">'
                     f'{_h(entry["status"])}</text>')
            continue
        last_t = native.trajectory.samples[-1].time_s
        if not (0 <= last_t <= horizon):
            raise ValueError("invalid native observation horizon")
        s.append(f'<line x1="{left}" x2="{left+last_t*scale:.7f}" '
                 f'y1="{y}" y2="{y}" stroke="{color}" stroke-width="3" '
                 'stroke-linecap="round"/>')
        alternative_count = 0
        for event in native.events:
            at = event.sample.time_s
            if not (0 <= at <= horizon):
                raise ValueError("event sample lies outside authored duration")
            alternative = at > last_t
            if alternative:
                # M24a may retain another terminal candidate with an
                # overlapping bracket slightly later than the selected stop.
                # Place its marker at the terminal BOUNDARY, never at the
                # later candidate's t: there is NO post-stop physical path.
                if native.trajectory.termination != "terminal-event":
                    raise ValueError("post-horizon nonterminal event")
                x = left + last_t*scale
                cy = y-8-alternative_count*8
                alternative_count += 1
                info = ('Overlapping terminal-boundary alternative, numerical '
                        'sample only; no post-stop physical motion')
            else:
                x, cy = left + at*scale, y
                info = "Located numerical root in the observed horizon"
            label = (f"{event.event_id}: {info}; candidate time {_fmt(at)} s; "
                     f"signed direction {event.direction:+d}; "
                     f"numerical bracket [{_fmt(event.time_bracket_s[0])}, "
                     f"{_fmt(event.time_bracket_s[1])}] s")
            alt_style = (' data-boundary-alternative="true" '
                         'stroke-dasharray="2 2"') if alternative else ''
            s.append(f'<circle cx="{x:.7f}" cy="{cy}" r="5" '
                     f'data-event-id="{_h(event.event_id)}"'
                     f'{alt_style} fill="{color}" stroke="#fff" stroke-width="1.5">'
                     f'<title>{_h(label)}</title></circle>')
        terminal = native.trajectory.termination
        x = left + last_t*scale
        if terminal != "duration":
            s.append(f'<rect x="{x-4:.7f}" y="{y-5}" width="8" height="10" '
                     f'fill="#b63838"><title>{_h(terminal)} '
                     f'at {_h(_fmt(last_t))} s</title></rect>')
    s.append("</svg>")
    return "".join(s)


def render_event_sweep(run: EventSweepRun, *,
                       title="M28b · Physical event comparisons") -> str:
    if type(run) is not EventSweepRun:
        raise TypeError("requires native EventSweepRun, not untrusted JSON")
    if (type(title) is not str or not 1 <= len(title) <= 120 or
            not title.isprintable() or title.strip() != title):
        raise ValueError("title must be printable trimmed text")
    data = run.as_dict()
    rows = _statuses(run)
    questions = run.plan.questions
    table = []
    for i, row in enumerate(rows):
        native = row["event_run"]
        for q in questions:
            qid = q.event_id
            if native is not None:
                results = native.as_dict()
                classification = results["outcomes"][qid]
                roots = [e for e in native.events if e.event_id == qid]
                last = native.trajectory.samples[-1].time_s
                root_details = "; ".join(
                    f"[{_fmt(e.time_bracket_s[0])}, {_fmt(e.time_bracket_s[1])}] s"
                    f" (t≈{_fmt(e.sample.time_s)} s; signed direction "
                    f"{e.direction:+d}; residual {_fmt(e.residual)} "
                    f"{e.residual_unit}; "
                    f"{_root_scope(e, last)})"
                    for e in roots)
                if not root_details:
                    root_details = ("No located root: not proven absent" if
                                    classification == "not-found-at-requested-resolution"
                                    else "No located root on retained horizon")
                finish = (f"{native.trajectory.termination} at "
                          f"{_fmt(native.trajectory.samples[-1].time_s)} s")
            else:
                classification, root_details, finish = (
                    "not-executed", "No native event solve", row["status"])
            shift = row["shifts"].get(qid)
            if shift is None:
                shift_label = "Baseline reference" if i == 0 else "Not comparable"
            elif shift["status"] == "one-resolved-pair":
                lo, hi = shift["numerical_bracket_shift_interval_s"]
                shift_label = (f"[{_fmt(lo)}, {_fmt(hi)}] s; numerical root estimate "
                               f"{_fmt(shift['estimated_numerical_shift_s'])} s")
            else:
                shift_label = f"Unpaired: {shift['reason']}"
            table.append("<tr><td>"+_h(row["id"])+
                         "<div class=\"muted\">"+_h(row["value"])+"</div></td><td>"+
                         _h(qid)+"</td><td>"+_h(classification)+
                         "</td><td>"+_h(root_details)+"</td><td>"+
                         _h(shift_label)+"</td><td>"+_h(finish)+"</td></tr>")
    raw = json.dumps(data, allow_nan=False, separators=(",", ":"))
    safe = (raw.replace("&", r"\u0026").replace("<", r"\u003c")
               .replace(">", r"\u003e"))
    baseline = run.plan.sweep.baseline
    all_ids = ", ".join(x for x, _ in baseline.source_annotations)
    source = f"Baseline origin: {baseline.origin}; authored provenance (unverified): {baseline.provenance}"
    policy = run.plan.policy
    details = (f"Shared typed questions: {', '.join(q.event_id for q in questions)}. "
               f"M24a max step {_fmt(policy.max_step_s)} s; root clock tolerance "
               f"{_fmt(policy.time_tolerance_s)} s. Original bound source IDs: {all_ids}.")
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>"""+_h(title)+"""</title>
<style>
:root{font:15px/1.5 system-ui,sans-serif;color:#172f46;background:#f4f6f9}
*{box-sizing:border-box}body{margin:0}header{background:#173b55;color:white;padding:18px max(14px,calc((100vw - 1180px)/2))}
h1{font-size:24px;margin:0 0 6px}header p{margin:0;font-size:13px;color:#d6e3ef}
main{max-width:1200px;margin:auto;padding:16px;display:grid;gap:14px}
section{background:white;border:1px solid #cfdae4;border-radius:8px;padding:16px;min-width:0;overflow-wrap:anywhere}
h2{font-size:17px;margin:0 0 6px}p{margin:5px 0 10px}
.muted{color:#5b6f80;font-size:12px;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed}
th,td{text-align:left;vertical-align:top;border-bottom:1px solid #e0e7ed;padding:8px;overflow-wrap:anywhere}
th{background:#eef3f7}th:nth-child(1){width:17%}th:nth-child(2){width:14%}
th:nth-child(3){width:15%}th:nth-child(4){width:21%}th:nth-child(5){width:19%}
.viewport{overflow-x:auto}svg{display:block;max-width:100%;width:100%;height:auto;border:1px solid #e0e7ed;background:#fcfdfe;border-radius:5px}
@media(max-width:520px){h1{font-size:19px}main{padding:10px}section{padding:10px}
table{min-width:720px}header{padding:14px}}
</style></head><body><header><h1>"""+_h(title)+"""</h1>
<p>Independently requalified typed events · native numerical roots · no browser ODE</p></header>
<main><section><h2>Observed event horizon</h2>
<p class="muted">Each line ends at its last native solved physical time.
Circles are located numerical event roots; red squares indicate
terminated or censored observation. Absence of a root marker is NOT proof
that an event never occurs.</p>"""+_timeline(rows, run)+"""</section>
<section><h2>Model and provenance</h2><p id="provenance">"""+_h(source)+"""</p>
<p class="muted" id="source-ids">"""+_h(details)+"""</p>
<p>Every scenario is a separate teacher-authored physical hypothesis from
the unchanged baseline, not a time-dependent change of a model parameter.
Potential references are not physical sweep axes. Provenance labels
are author assertions, not verification of a textbook scan.</p></section>
<section><h2>Every detected event and numerical shift</h2>
<p class="muted">Brackets and shifts are localization bounds for the
numerically integrated paths, not exact ODE error enclosures. An event is
paired only when exactly one resolved event exists in each case. Multiple
crossings, unresolved roots, stop-surface observations and source guards are
not silently reduced to a single arbitrary time or assumed contact.</p>
<div class="viewport"><table><thead><tr><th>Scenario / value</th>
<th>Physical question ID</th><th>Finite-resolution outcome</th><th>Actual roots and brackets</th>
<th>Eligible root-time shift</th><th>Run termination</th></tr></thead><tbody id="event-rows">"""+ "".join(table)+"""</tbody></table></div></section>
<section><h2>Interpretation limits</h2>
<p>One parameter was changed per separate model, retaining the original
source identities and numerical event policy. The chart is an event-time
summary, not source apparatus geometry, a particle motion path, or an event
computed in the browser. Detectors exert no force; a terminal observation
is not contact, absorption or electrical induction. All displayed values
came from separately validated and integrated native Python scenarios.</p>
</section></main><script type="application/json" id="m28b-evidence">"""+safe+"""</script></body></html>"""


def write_event_sweep_html(run: EventSweepRun, output, **kwargs) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_event_sweep(run, **kwargs), encoding="utf-8")
    return path
