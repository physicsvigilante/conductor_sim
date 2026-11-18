"""M36e R2: independently validate the ACTUAL embedded 65x65 solid-sphere grid.

This is standard-library-only Gauss-law arithmetic, not another production
solver. It checks EVERY field/potential value against source/teacher geometry,
net charge and explicit plane coordinates, independently of M5 callbacks.
Grid, physical answer cards and the detached native manifest are separate
evidence channels: successful equality among them is not a cryptographic
signature against an attacker who controls both source and validators.
"""
from __future__ import annotations

from math import hypot, isclose, isfinite, pi

EPSILON0_SI = 8.8541878128e-12
K = 1.0 / (4.0 * pi * EPSILON0_SI)
N = 65


def demand(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def close(value, expected, label, *, rel=3e-9, abs_tol=1e-8):
    demand(type(value) in (int, float) and isfinite(value),
           "source grid has invalid/nonfinite value: " + label)
    demand(isclose(value, expected, rel_tol=rel, abs_tol=abs_tol),
           f"embedded grid SI mismatch {label}: saved {value!r}, expected {expected!r}")


def vector(actual, expected, label):
    demand(isinstance(actual, (list, tuple)) and len(actual) == 3,
           "embedded grid has missing physical field/plane 3-vector: " + label)
    for component, (value, reference) in enumerate(zip(actual, expected)):
        close(value, reference, f"{label}[{component}]")


def check_embedded_solid_grid(manifest: dict, bundle: dict) -> dict:
    """Check EVERY actually saved frame potential, signed E and metal region."""
    demand(bundle["schema"] == "conductor_sim.visualization.v1",
           "wrong embedded physical-grid schema")
    states, frames = manifest["native_states"], bundle["frames"]
    demand(len(states) == len(frames) == 3,
           "exactly three real original/teacher solid frames required")
    verified = []
    for i, (state, frame) in enumerate(zip(states, frames)):
        g = state["physical_geometry"]
        center = g["center_m"]
        R = g["radius_m"]
        q = state["observed_conductor_results"][0]["net_charge_c"]
        demand(state["physical_shape_type"] == "SolidSphere"
               and g["inner_radius_m"] == 0,
               "grid verifier requires genuine single SOLID source")
        demand(type(R) in (float, int) and isfinite(R) and R > 0,
               "invalid physical source solid radius")
        demand(type(q) in (float, int) and isfinite(q),
               "invalid physical source solid charge")
        demand(frame["label"] == state["state_title"],
               "wrong original/teacher frame labelled as source")
        demand(len(frame["geometry"]) == len(frame["conductor_results"]) == 1,
               "extra fake physical conductor in embedded grid")
        geom, conductor = frame["geometry"][0], frame["conductor_results"][0]
        demand(geom["name"] == conductor["name"] == state["physical_source_id"],
               "grid physical source identity changed")
        demand(geom["kind"] == "sphere" and geom["visible"] is True,
               "grid geometry not a real physical circular metal")
        close(geom["inner_radius_m"], 0, f"solid inner radius {i}", abs_tol=0)
        close(geom["outer_radius_m"], R, f"real solid outer radius {i}", abs_tol=1e-14)
        close(conductor["net_charge_c"], q, f"grid conductor charge {i}", abs_tol=1e-20)
        close(conductor["outer_surface_charge_c"], q,
              f"grid conductor outer face charge {i}", abs_tol=1e-20)
        close(conductor["potential_v"], K*q/R, f"grid conductor equipotential {i}")
        demand(frame["source_masks"] == [], "fake point source mask on solid")
        demand(len(frame["profiles"]) == 1
               and frame["profiles"][0]["surface"] == "outer",
               "fake induced inner charge-density profile in solid grid")

        plane = frame["plane"]
        origin, u, v = plane["origin_m"], plane["u"], plane["v"]
        vector(origin, origin, f"plane origin {i}")
        vector(u, u, f"plane u {i}")
        vector(v, v, f"plane v {i}")
        # Explicit orthonormal meridian to rule out arbitrary fake world axes.
        close(sum(x*x for x in u), 1, f"plane u unit {i}")
        close(sum(x*x for x in v), 1, f"plane v unit {i}")
        close(sum(a*b for a, b in zip(u, v)), 0,
              f"plane u/v orthogonality {i}", abs_tol=1e-12)
        scene = frame["grid"]
        xs, ys = scene["x_m"], scene["y_m"]
        potentials = scene["potential_v"]
        fields = scene["electric_field_v_per_m"]
        regions = scene["region"]
        demand(len(xs) == len(ys) == len(potentials) ==
               len(fields) == len(regions) == N,
               "actual 65x65 physical grid missing")
        demand(all(xs[j] < xs[j+1] for j in range(N-1))
               and all(ys[j] < ys[j+1] for j in range(N-1)),
               "source physical grid coordinate order altered")
        expected_geometry_center = (
            sum((center[k]-origin[k])*u[k] for k in range(3)),
            sum((center[k]-origin[k])*v[k] for k in range(3)),
        )
        demand(len(geom["center_m"]) == 2, "source 2D metal center missing")
        for k in range(2):
            close(geom["center_m"][k], expected_geometry_center[k],
                  f"embedded center projection state={i},axis={k}")
        metal, vacuum, boundary = 0, 0, 0
        for iy, y in enumerate(ys):
            demand(len(potentials[iy]) == len(fields[iy]) ==
                   len(regions[iy]) == N,
                   "physical grid row is truncated")
            for ix, x in enumerate(xs):
                world = tuple(origin[k] + x*u[k] + y*v[k] for k in range(3))
                delta = tuple(world[k] - center[k] for k in range(3))
                r = hypot(*delta)
                is_surface = isclose(r, R, rel_tol=1e-12, abs_tol=1e-14)
                potential = K*q/(R if r <= R else r)
                close(potentials[iy][ix], potential,
                      f"state={i} potential x={ix} y={iy}")
                region, electric = regions[iy][ix], fields[iy][ix]
                if is_surface:
                    # Ideal E has two unequal one-sided limits at metal.
                    demand(region == "surface" and electric is None,
                           f"embedded grid metal boundary falsely has single E state={i} xy={ix},{iy}")
                    boundary += 1
                elif r < R:
                    demand(region == "metal",
                           f"embedded grid incorrectly paints solid interior state={i} xy={ix},{iy}")
                    vector(electric, (0., 0., 0.),
                           f"metal interior field state={i} xy={ix},{iy}")
                    metal += 1
                else:
                    demand(region == "vacuum",
                           f"embedded grid falsely labels outside metal state={i} xy={ix},{iy}")
                    vector(electric, tuple(K*q*d/(r**3) for d in delta),
                           f"exterior signed E state={i} xy={ix},{iy}")
                    vacuum += 1
        demand(metal > 0 and vacuum > 0,
               "source grid never sampled both REAL solid and surrounding vacuum")
        verified.append({
            "state_index": i,
            "original_or_teacher_role": state["case_role"],
            "real_independent_SI_grid_points": N*N,
            "metal_points": metal,
            "vacuum_points": vacuum,
            "surface_jump_points": boundary,
        })
    return {
        "schema": "conductor_sim.m36e_full_embedded_grid_gauss.v1",
        "states": verified,
        "actual_checked_potential_and_E_points": N*N*3,
        "source_original_apparatus_increment": 0,
        "uses_only_standard_library_SI": True,
    }
