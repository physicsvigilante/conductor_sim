# M2b — centred charge and grounded spherical shell

**Scope:** one 3D spherical conducting shell and zero or more charges **exactly at its center**, combined into one net source. The shell can be `Grounded()` or `FixedNetCharge(Q)`; the field and conductor results come from **actual FEMM 5 solves**, not from the analytical solver. M2a's empty-shell cases stay supported.

## Physics, not hardcoded problem cases

Given inner radius `a`, outer radius `b`, source charge `q`, and shell net charge `Q_shell`, Gauss's law gives `Q_inner=-q`, `Q_outer=Q_shell+q`. For an isolated shell `Q_shell` is fixed; for a grounded shell FEMM determines its charge while enforcing `V_shell=0`. With a centered source and no other charges, the grounded solution has `Q_shell=-q`, `Q_outer=0`. The analytical backend remains an independent check, never a replacement for FEMM data.

## Why the numerical source is a tiny *charged sphere*

A naive FEMM axisymmetric source at nonzero cylindrical radius would revolve around the axis and would **not** represent an ordinary 3D point charge. In M2b, the compiler instead introduces a concentric numerical core of radius `source_radius_fraction * a` (default `0.10*a`) and assigns it a separate FEMM conductor with *fixed total charge* `q`. By spherical symmetry, at any radius greater than this core, the ideal point source and tiny charged sphere have the same electrostatic solution in this arrangement. This is a numerical regularization, **not a new component in the authored problem**.

The core is completely separate from the physical shell. FEMM directly reports its net charge; the shell's inner/outer surface-charge totals are inferred from Gauss's law and the FEMM-reported source and shell totals. They are **not independently measured surface-density distributions**.

The numerical solution rejects potential and field requests at radii inside/on the artificial core; it also rejects field samples directly on conductor surfaces. The core radius is exposed through `solution.source_radius_m`, and `solution.numerical_source_charge_c` comes directly from FEMM. No numerical source core is created for net-zero centered charges.

## Numerical acceptance criteria

The *separate* real-FEMM workflow runs the existing M2a benchmarks plus new grounded, isolated-neutral, nonzero-shell-charge, positive/negative source, empty-grounded, and source radius/mesh-sensitivity tests. Comparisons check shell charge/potential, source charge, sampled cavity/metal/exterior potential and field. The tests explicitly check that grounded exteriors are near zero and that the metal has negligible field. The finite computational domain has a monopolar Robin boundary; model accuracy and variation under source-radius changes are checked at safe, non-singular sample points.

Passing these tests validates **only** this constrained spherical geometry and the documented tolerance levels. It is not evidence for off-center charge distributions, genuinely 3D asymmetry, arbitrary conductors, direct local surface-charge densities, production process isolation or visualization.

## Run

Set `FEMM5_BINDING` to the real FEMM 5 `pyfemm/femm.py`; set `FEMM_DIR` to the directory with `femm_headless` and `belasolv` if needed. Then:

```bash
python examples/femm5_centered_charge.py
python -m pytest tests/test_femm5_live.py -vv -rA
```

The ordinary GitHub Python workflow does *not* run FEMM; its integration tests are skipped without the real binding. The dedicated workflow builds a pinned official release and executes the live tests with FEMM present.
