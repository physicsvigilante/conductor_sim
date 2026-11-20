# M41a — Frozen cohort and scientific evidence charter

Status: PREIMPLEMENTATION CHARTER. Tracking issue #106. This exact file is committed by itself before M41a source/scripts/tests, on a separate branch from RELEASED, independently reviewed audit main commit 337001f2e0d1ce62c87fa4352993c4a074cfbde6 / native Git TREE 25e91e0fd53749bf6c6c578583310b219e1ecb4f. Do not rebase M41a onto unmerged M40c PR #105 or infer M40c is merge-approved.

## Frozen source cohort, stable IDs, questions and physical family

Source A = 02_conductors_half_size.pdf, 17 scanned pages, SHA256 9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d.
Source B = POTENTIAL_CONDUCTORS1.pdf, 71-page teaching notes, SHA256 8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5.

14 *originally existing M36a anchor IDs*, registered before new numerical runs or successes are selected:

| ID | Source/page | Printed item | Physical claim to test |
| --- | --- | --- | --- |
| p02-single-15 | A p2 | Single Correct Q15 | far-separated, wire-connected spherical conductors (leading approximation, NOT a general near-contact solver) |
| p02-single-19 | A p2 | Single Correct Q19 | electron reversal near charged ideal conducting sheet |
| p03-single-27 | A p3 | Single Correct Q27 | charge/energy release when nested solid sphere is wired to enclosing metal shell |
| p03-single-30 | A p3 | Single Correct Q30 | electron voltage-gap transit, actual elapsed-time motion |
| p05-single-47 | A p5 | Single Correct Q47 | rigid dipole orientation in uniform field, compare potential energies |
| p08-multiple-21 | A p8 | Multiple Correct Q21 | positive charge PRESCRIBED to approach grounded sphere at constant small speed; static induced Q and conventional wire-current direction/magnitude |
| p11-linked-28 | A p11 | Linked Comprehension Q28 | IRREGULAR outer conductor, two enclosed cavities: Gauss total only, uniformity of exterior induced charge NOT established |
| p12-linked-40 | A p12 | Linked Comprehension Q40 | two rings, transverse E: endpoint work; not a proof of actual straight-line dynamics asked by different Q41 |
| p15-archive-02 | A p15 | Archive IIT-JEE 2012 Q2 | proton initially between vertical plates, electric + gravity 45-degree acceleration |
| jee2021-p1-q09-q10-equipotential | A p16 | Paper 1 Paragraph Q3–4, official JEE Advanced 2021 2-point-charge apparatus Q9–10 | V=0 circle, centre and radius in metres, count ONE physical system |
| p26-conductor-potential | B p26 | Illustration 1 | true SOLID conducting sphere at r=R/2 and r=3R/2; exclude nonconducting sphere Illustration 2 |
| p61-external-grounded-sphere | B p61 | Illustration 1 | final STATIC net charge appearing on a grounded sphere near external +Q; source symbolic, teacher inputs must be explicit |
| p69-cavity-central-vs-shifted | B p69 | Illustration 7 | central vs off-centre enclosed source in spherical cavity; same inner whole-face Q, different local density, outer Q unchanged |
| p70-cavity-offset | B p70 | Illustration 8(i)/(ii) | genuinely irregular cavity in (i) and irregular exterior in (ii); local sigma/exterior uniformity cannot be substituted with a qualified spherical model |

The source pages were visually inspected as rendered PNGs from exact original PDF bytes; workbook PDF text layer is empty. These are not automatically official JEE exam problems: workbook exercise and notes Illustration labels are retained as source-native provenance. The source-labeled official 2021 apparatus is one grouped system; official examination question identity beyond this PDF is not independently authenticated here.

## Evidence status contract

Store independent fields PER registered anchor:
- source SHA and physical page, printed anchor, source type/provenance, and original-wording/diagram check;
- model and qualification, exact original GIVEN and teacher CHOSEN quantities separately, and explicit unsupported physics;
- observed actual *newly executed* native output from the released solver (file/hash) plus independent source-free physical reference (file/hash), never a status inferred from an API doc;
- HTML artifact existence separately from actual JavaScript-enabled, disabled and mobile Chromium verification; do NOT claim browser pass on HTML existence alone;
- original textbook answer completeness requiring actual given values/asked observable, never inferred from generic physics agreement;
- educational usefulness = NOT_EVALUATED_WITH_STUDENTS unless a separate actual teacher/student study is conducted.

Allow measured outcomes: verified_scoped_native, partial, unsupported_source_geometry, not_run, failed. Every source must have exactly one row and audit-frozen source ID; original answer completeness and rendered-vs-HTML-only have their own fields, NOT inferred from native success. Fractions/percentages must show the scope and denominator; 14 source anchors include 2 deliberate unsupported controls and NOT 14 official JEE question successes.

Expected pretrial routing (NOT PRECERTIFIED RESULTS): reuse released M36b source/native+independent reference for IDs p02-15, p02-19, p03-30, p05-47, p12-40, p15-02, jee2021 group; M36c for p03-27 and p69; M36d for notes p26; M37b for p08; source-specific p61 static Kelvin may be newly implemented with honest teacher numbers; p11/p70 should refuse unsupported original local geometry. No new solver law, arbitrary 3D conductor or M40c source relocation assumption.

## Release safeguards

Preserve historical M21a/M36a inventories and two PDFs unchanged. Do not add a universal benchmark/scene graph infrastructure. Execute reference and native source runs on source-pinned checkout with actual PDF hashes; enforce physical and source-attribution refusal controls, truthful statuses and distinct denominator. Run Python 3.10/3.12 and inherited true FEMM/Chromium CI before one focused independent audit of factual source interpretation and honest evidence/answer coverage. M41a PR remains DRAFT/UNMERGED until independent PASS and **separate explicit owner MERGE approval**, exact-head safeguard, native reviewed tree match on main and full actual postmerge checks. Never use a user's generic 'continue' as approval to merge M40c PR #105.
