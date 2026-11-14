# M21a — JEE Advanced-focused, source-anchored coverage audit charter

## Priority

**JEE Advanced educational relevance and honest physical fidelity determine
the next physics milestone.** Engine composability and LLM authorability are
enablers, not stand-alone reasons to add an abstract framework or showcase.
The task is an audit and prioritization, **not** a conductor solver extension.

## Fixed sources

1. `02_conductors_half_size.pdf` — 17 scanned PDF pages, original file pinned
   by M13c SHA-256 `9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d`.
   A mixed workbook containing conductor, potential, ring, dynamics and
   explicitly printed archived exam questions. Printed provenance is not
   independent exam certification.
2. `POTENTIAL_CONDUCTORS1.pdf` — 71 pages, pinned by M6a SHA-256
   `8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5`.
   Teaching illustrations, **not assumed to be official JEE Advanced past-paper questions**.

The audit must inspect actual PDF pages/figures, not only inherit old case
descriptions. Render scanned pages; extract and crosscheck searchable PDF page
text. Recheck displayed charge signs, geometry, source roles and constraints.
No OCR unless all other routes fail; uncertainty from an unreadable figure is
an explicit limit.

## Preselected primary cohort

Assess **all 18** previously anchored scanned-workbook tasks, preserving
their PDF page/printed question identity (M13c):

- p01-single-08; p02-single-15; p02-single-19;
- p03-single-27; p03-single-29; p03-single-30;
- p05-single-47; p07-multiple-13; p08-multiple-21;
- p09-multiple-27; p10-linked-13; p10-linked-14;
- p11-linked-28; p12-linked-40; p13-matrix-05;
- p14-numerical-07; p15-archive-02; p16-archive-numerical-01.

Assess **11** selected complementary teaching illustrations from M6a:
- p13-projected-charge; p20-ring-axis; p25-insulating-sphere;
- p26-conductor-potential; p27-cavity-two-shells;
- p29-outer-charge-change; p61-external-grounded-sphere;
- p68-connected-separated-spheres; p69-cavity-central-vs-shifted;
- p70-cavity-offset; p70-three-cavities.

The 29-anchor cohort is deliberately selected, not statistically random or
exhaustive. It should contain simple controls, actual labelled exam archives,
intermediate multi-conductor problems and truly unsupported arrangements.

## Classification and factual constraints

For every item record:
- source filename + pinned SHA-256 + physical PDF page + printed question;
- provenance level: **printed archive label / workbook exercise /
  teaching derivation**, independent original-paper verification separately;
- short physical system description anchored in the figure and text;
- JEE Advanced relevance: high / medium / foundational-control, with rationale;
- scope: whether the **exact** question, a subset of observables, or an
  explicitly modified teacher hypothetical can be represented;
- analysis type: static equilibrium / precomputed static states /
  genuine initial-value motion / externally prescribed sources;
- current existing APIs/backends with specific evidence links;
- capability classification:
  A = already representable and suitably simulatable;
  B = underlying physics available but a small **composable adapter** or
      authoring/viewer capability is needed for fidelity;
  C = genuinely new qualified physics/model relationship required;
  D = currently outside project scope or the source cannot be interpreted;
- distinguish a **certified execution** from an evidence-based engineering
  assessment. Never score a hypothetical substitute as full fidelity.
- blockers/limitations and a JEE-focused next-milestone recommendation.

The priority score is qualitative and **must not** treat duplicate anchors,
introductory examples or selective PDF-page sampling as occurrence frequencies
in JEE Advanced examinations. No statistical claim about exam coverage.

## Acceptance

1. Exact original PDF SHA and page count verified by source inspection.
2. Figures/geometry checked for the high-priority scanned cases, especially
   connected conductors, cavity charges, ring fields and exam archive labels.
3. A machine-readable 29-case matrix plus human-readable evaluation, grouped
   by useful *problem families*, not just components or implementation stages.
4. Independently re-exercise a small set of existing concrete authorship
   pipelines, and provide explicit verified vs assessed distinction.
5. Identify **one** next engineering milestone justified by JEE relevance,
   composability, fidelity, expected reach and implementation burden.
6. No changes to the production solver/renderer; no fabricated physics, and
   no source-question answer-completeness claims without supporting evidence.
7. Prefer **one narrowly scoped CI verification workflow**, not a new generic
   scenario manager, full extra FEMM build or heavy framework.
