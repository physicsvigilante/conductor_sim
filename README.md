# conductor_sim

Scientifically qualified, LLM-authorable **teaching simulations for JEE electrostatics**. The goal is faithful physical interpretation and useful exploration—not automatic reproduction of every textbook answer.

**Released baseline: M40b**, main merge `4bae2f6be5a1689b9e1dc80d2dd146b77d366a3e`. See the [current capability and verification map](docs/current_capabilities.md) for qualified domains, release evidence and limitations. The [post-M40b project audit](docs/post_m40b_audit.md) is a separate correction candidate; **M40c remains charter-only** in [issue #103](https://github.com/physicsvigilante/conductor_sim/issues/103).

## Start with a teaching question

| Question | Existing route |
| --- | --- |
| What are the field, potential, force and energy near a qualified conductor? | Typed `ElectrostaticProblem`, analytical backends and optional FEMM; [architecture](docs/architecture.md), [force/energy examples](examples/force_energy.py). |
| How does a test charge move in a supported fixed-source arrangement? | Prescribed fields, native motion/event observers and source guards; [prescribed-source distinction](docs/m19a.md), [structural source reuse](docs/m22a.md). |
| How does a physical revision change a verified observation? | Retained `PrescribedScenario`, source identities, sweeps and optional graph/fork inquiries; [multi-stage exploration](docs/m35a.md). |
| Which existing examples are tied to original JEE source material? | [M36b](docs/m36b.md), [M36c](docs/m36c.md), [actual solid-sphere refinement](docs/m36d.md), [constant-speed grounding current](docs/m37b.md). Source givens and teacher choices stay separate. |
| Why can a neutral cavity face have nonzero local induced charge density? | [M39a whole-face Gauss totals](docs/m39a.md), [M40a spherical local sigma](docs/m40a.md), [M40b offline geometry and point atlas](docs/m40b.md). |

## Generate an offline example

From the repository root, using Python 3.10 or 3.12 in an isolated environment:

```sh
python -m pip install -e ".[test]"
python -m tools.m40b_spherical_atlas_example --output-dir output/m40b
python -m tools.m40b_spherical_atlas_reference --input output/m40b/m40b-teacher-atlas.json
```

Open `output/m40b/m40b-teacher-atlas.html` locally. This is a teacher-authored static spherical-cavity example: actual XY projections with explicit Z coordinates, signed surface **point** samples, and separate isolated/grounded equilibria. It is not a general 3D field map, a grounding transient, or a newly solved original workbook question.

For the source-backed prescribed approach example:

```sh
python -m tools.m37b_source_approach --output-dir output/m37b
python -m tools.m37b_source_reference --input output/m37b/p8-ground-current.json
```

A prescribed constant-speed source is not a freely accelerating particle. The generated page identifies the original source facts and teacher numerical choices.

## Validation and authoring boundaries

```sh
python -m pytest -m "not femm5" -q
```

This excludes actual FEMM tests; it must not be reported as FEMM numerical validation. Hosted release checks additionally cover supported Python versions, real FEMM, independent scientific references and real Chromium. Candidate fixes need their own source-pinned review, explicit merge approval and post-merge checks.

Models are reusable **within their declared physical and numerical regimes**. Every revision is requalified; a valid baseline does not authorize every change. Keep responding conductors separate from prescribed-source superposition, physical time separate from display playback, and whole-face charge separate from sampled local density. See the [composability contract](docs/composability_contract.md) and [five-dimension evaluation contract](docs/evaluation_contract.md).

Unsupported irregular 3D conductor geometry, arbitrary coupled dynamics, or incomplete original-question interpretations are refused or explicitly labeled partial—not silently replaced by an easier system. JSON consistency is not cryptographic authentication of independently edited HTML or historical teacher input.

## Historical milestone reference

The previous long README is preserved **byte-for-byte**, with its root-relative links intact, in [README-MILESTONE-HISTORY.md](README-MILESTONE-HISTORY.md). Its old release/draft wording is historical. Use the [current map](docs/current_capabilities.md), not that snapshot or the package's historical `0.1.0`, to identify the released scope.
