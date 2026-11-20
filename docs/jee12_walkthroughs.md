# JEE-12 — two original-workbook answer walkthroughs

Issue #110 is intentionally limited to two original workbook questions. The [preimplementation charter](jee12_walkthrough_charter.md) was committed before the production source. Both offline HTML/SVG pages are static and fully usable when JavaScript is disabled.

## Run

```sh
python -m pip install -e ".[test]"
python -m tools.jee12_walkthroughs --output-dir output/jee12-walkthroughs
python -m tools.jee12_walkthrough_reference --input-dir output/jee12-walkthroughs
```

Open:
- `output/jee12-walkthroughs/p1-external-work.html`
- `output/jee12-walkthroughs/p3-solid-shell-energy.html`

The original PDF `02_conductors_half_size.pdf` is frozen by SHA-256 `9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d`. It is a **workbook**: individual exercises are not automatically official JEE examination items.

Original PDF p1 Single Correct Q8: +10 V/m uniform x-field, electron A→B, 5-3-4 displacement. Native `endpoint_field_work` yields +40 eV external work **by us** and −40 eV field work, printed option (2). A separate teacher rerun doubles E to +20 V/m and yields +80 eV; no invented free electron trajectory or elapsed time.

Original PDF p3 Single Correct Q27: truly filled solid sphere R=.10 m carrying +20 µC inside a concentric uncharged shell R=.20 m. Released `ConnectedShellBackend` computes authentic before/after **static** equilibria and a finite-shell field energy decrease ~8.98751 J. Textbook zero-thickness limit ~8.98755 J rounds to option (2) **9 J**; 1 µm shell width is a disclosed **teacher numerical geometry**, not printed. A separate Q=10µC rerun has one-quarter the release; no time waveform, wire-current or rate-of-heat model.

The standalone source/provenance checker independently computes SI signs, charges, conductor faces and energies with Python's standard library only. The browser acceptance inspects genuine static generated diagrams and data in real desktop and 375px mobile Chromium with JS on/off. Both are distinct from cryptographically authenticated external teacher histories. This development does not modify the JEE-12 original-answer certification ledger or resolve legacy viewer issue #111.

**Release boundary:** focused green Python 3.10/3.12 and Chromium checks, full inherited regression suite, one narrow independent source-answer review, explicit owner merge approval, guarded exact-tree merge and true postmerge CI.
