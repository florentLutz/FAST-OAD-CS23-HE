# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
run_ampere_case.py
===================
Runs the ONERA AMPERE hybrid PEMFC+H2+battery case (40 EDF / 10 clusters):
  - data/generated/assembly_ampere_final.yml
  - data/generated/configuration_ampere_final.yml
  - results/generated/ampere_final_in.xml

16 battery modules per cluster and a 70% battery/PEMFC power split (see configuration_ampere_
final.yml) are the validated defaults; wing sizing uses fastga_he.loop.wing_area (DEP-equilibrium,
slipstream-aware). The airframe (fuselage, landing gear, tail arrangement) is Pipistrel-derived;
only the propulsion system and the wing/tail/TLAR parameters are AMPERE's real values (Dillinger,
Ridel & Doll, ICAS 2018-0492). See make_report.py's generated report for the full real-vs-simulated
comparison and known limitations.

Usage (from integration_tests/ampere_final/):
    python run_ampere_case.py
    python run_ampere_case.py --battery-modules 14   # re-tune the battery for a different mass
    python run_ampere_case.py --power-split 60       # re-tune the battery/PEMFC split

After a successful run, generate the figures + report with:
    python make_report.py
"""

import argparse
import datetime
import os
import sys

import numpy as np

import fastoad.api as oad
from openmdao.core.analysis_error import AnalysisError
from fastoad.openmdao.exceptions import FASTNanInInputsError

_parser = argparse.ArgumentParser(add_help=True, description=__doc__)
_parser.add_argument(
    "--battery-modules",
    type=float,
    default=16.0,
    help="number_modules seeded per battery_pack_N cluster (default 16, the validated value).",
)
_parser.add_argument(
    "--power-split",
    type=float,
    default=70.0,
    help="data:propulsion:he_power_train:DC_splitter:dc_splitter_N:power_split (percent) seeded "
    "for all 10 clusters -- share of propulsive power routed through the battery branch at every "
    "mission point (default 70, the validated value).",
)
_args, _ = _parser.parse_known_args()


class _Tee:
    """Writes to every given stream at once, so OpenMDAO's own print() calls (solver
    residuals, stall/convergence messages) end up in both the terminal and the log file
    without changing how the rest of the script prints things."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


CASE_NAME = "ampere_final"
CONFIG_YML = os.path.join("data", "generated", f"configuration_{CASE_NAME}.yml")
INPUT_XML = os.path.join("results", "generated", f"{CASE_NAME}_in.xml")
BATTERY_MODULES_PER_CLUSTER = _args.battery_modules
POWER_SPLIT_PERCENT = _args.power_split

# Real AMPERE MTOW target is 2400 kg (Dillinger et al., ICAS 2018-0492, Table 1).
MTOW_SEED_KG = 2400.0
OWE_SEED_KG = 1700.0


def _print_nan_outputs(problem):
    nan_vars = [
        var.name
        for var in oad.VariableList.from_problem(problem, io_status="outputs")
        if var.val is not None
        and any(str(v) == "nan" for v in getattr(var.val, "flatten", lambda: [var.val])())
    ]
    if nan_vars:
        print(f"\n{len(nan_vars)} output(s) contain NaN after the failed run:")
        for name in nan_vars[:40]:
            print(f"  {name}")
    else:
        print("\nNo NaN found in outputs (failure is likely a non-convergence, not a NaN blow-up).")


def _enable_equilibrium_residual_printing(problem):
    """Turns on iprint=2 (prints ||R|| at every Newton iteration, not just stall/failure
    warnings) for the compute_dep_equilibrium solver -- the one that solves alpha/thrust/delta_m
    at each mission point. Falls back to listing every pathname containing "equilibrium" if the
    expected one isn't found, since the exact nesting can differ slightly between environments."""

    target = None
    for system in problem.model.system_iter(include_self=True, recurse=True):
        if system.pathname.endswith("solve_equilibrium.compute_dep_equilibrium"):
            target = system
            break

    if target is not None:
        target.set_solver_print(level=2, type_="NL")
        print(f"[log] iprint=2 enabled at: {target.pathname}\n")
    else:
        print(
            "[log] Could not find 'solve_equilibrium.compute_dep_equilibrium' -- pathnames "
            "containing 'equilibrium' found in the model:"
        )
        for system in problem.model.system_iter(include_self=True, recurse=True):
            if "equilibrium" in system.pathname:
                print(f"  {system.pathname}")
        print()


def _fill_landing_control_params_from_mission():
    """
    Only matters because this case's config has wing_area: id: fastga_he.loop.wing_area
    (UpdateWingAreaLiftDEPEquilibrium) instead of the plain fastga.loop.wing_area. That submodel
    exposes an independent "_landing" copy of every power-train "control parameter"
    (get_control_parameter_list() in powertrain.py +
    zip_equilibrium_input()'s renaming in wing_area_cl_dep_equilibrium.py) -- e.g.
    ...:DC_splitter:dc_splitter_1:power_split_mission also gets a ...:power_split_landing input --
    so the wing can be sized against a landing-point equilibrium that's independent of whatever
    control values the mission itself uses. These "_landing" variables have no value anywhere in
    the case's input XML (they didn't exist before wing_area: was switched away from the plain
    fastga.loop.wing_area), so write_needed_inputs() leaves them at NaN.

    For anything ending in "_landing" that's still NaN, fall back to its "_mission" sibling's
    representative (median) value -- landing is a single flight point anyway. The rename in
    zip_equilibrium_input() is "replace '_mission' with '_landing'" when "_mission" appears in
    the name, else "append '_landing'" (e.g. thrust_distribution -> thrust_distribution_landing)
    -- so both candidate original names are tried since which rule applied isn't recoverable
    from the "_landing" name alone. Variables found via the "replace" rule are shape_by_conn=True
    per-mission-point arrays and get collapsed to shape (1,) (the landing equilibrium always
    runs at number_of_points=1) even if already non-NaN but wrongly-shaped; variables found via
    the "append" rule are per-propulsor arrays and are left at their own shape.
    """
    if not os.path.exists(INPUT_XML):
        return

    datafile = oad.DataFile(INPUT_XML)
    names = set(datafile.names())
    filled = []

    for name in sorted(names):
        if not name.endswith("_landing"):
            continue

        var = datafile[name]
        current = np.asarray(var.value, dtype=float)

        base = name[: -len("_landing")]
        replace_rule_name = base + "_mission"
        append_rule_name = base
        if replace_rule_name in names and replace_rule_name != name:
            mission_name = replace_rule_name
            is_per_point = True  # shape_by_conn, must end up shape (1,)
        elif append_rule_name in names and append_rule_name != name:
            mission_name = append_rule_name
            is_per_point = False  # e.g. thrust_distribution_landing, per-propulsor, leave shape
        else:
            continue

        needs_fill = np.any(np.isnan(current))
        needs_reshape = is_per_point and current.size != 1
        if not (needs_fill or needs_reshape):
            continue

        mission_val = np.asarray(datafile[mission_name].value, dtype=float)
        if np.all(np.isnan(mission_val)):
            continue

        representative = float(np.nanmedian(mission_val))
        if is_per_point:
            var.value = np.array([representative])
        else:
            var.value = np.full(np.shape(mission_val), representative)
        filled.append((name, mission_name, representative))

    if filled:
        datafile.save()
        for name, mission_name, val in filled:
            print(f"{name}: filled with {val:.3g} (representative value from {mission_name}).")


def _run(log_path):
    print(f"=== Case: {CASE_NAME} (40 EDF / 10 clusters, hybrid PEMFC+H2+battery) ===")
    print(f"Full log for this run: {log_path}\n")

    if not os.path.exists(INPUT_XML):
        raise FileNotFoundError(
            f"{INPUT_XML} not found -- make sure you're running this from "
            "integration_tests/ampere_final/."
        )

    _fill_landing_control_params_from_mission()

    configurator = oad.FASTOADProblemConfigurator(CONFIG_YML)
    try:
        problem = configurator.get_problem(read_inputs=True)
    except FASTNanInInputsError as e:
        raise RuntimeError(
            f"\n{INPUT_XML} still has NaN in some input -- {e}\n"
            "Run list_inputs()/VariableList.from_problem again to find which variable changed "
            "(could be a different submodel active in your environment vs. the one this was "
            "built with)."
        )

    problem.setup()
    _enable_equilibrium_residual_printing(problem)

    # Initial MTOW-loop seed.
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=MTOW_SEED_KG)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=OWE_SEED_KG)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=MTOW_SEED_KG)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=MTOW_SEED_KG)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=MTOW_SEED_KG)

    # data:aerodynamics:cruise:mach/unit_reynolds and their low_speed counterparts are internal
    # (non-XML) variables computed by comp_unit_reynolds from TLAR:v_cruise/v_approach + cruise
    # altitude. They're declared with val=np.nan in FAST-GA, and since aero_vlm runs under
    # NonlinearRunOnce (no internal iteration), on the very first NLBGS pass every downstream
    # consumer sees them still NaN if comp_unit_reynolds hasn't executed yet. XfoilPolar isn't
    # robust to that (mach=nan/reynolds=nan -> malformed 1-row polar -> TypeError, a hard crash).
    # Seeding sane physical values here (from v_cruise=150kn/8000ft and v_approach=58.5kn/sea
    # level via ISA) avoids NaN on that first pass; comp_unit_reynolds overwrites them with the
    # real computed values on this and every subsequent NLBGS iteration regardless.
    problem.set_val("data:aerodynamics:cruise:mach", val=0.233)
    problem.set_val("data:aerodynamics:cruise:unit_reynolds", units="m**-1", val=4.34e6)
    problem.set_val("data:aerodynamics:low_speed:mach", val=0.0885)
    problem.set_val("data:aerodynamics:low_speed:unit_reynolds", units="m**-1", val=2.06e6)

    print(f"Battery modules per cluster (fixed seed): {BATTERY_MODULES_PER_CLUSTER}\n")
    for cluster in range(1, 11):
        problem.set_val(
            f"data:propulsion:he_power_train:battery_pack:battery_pack_{cluster}:number_modules",
            val=BATTERY_MODULES_PER_CLUSTER,
        )

    print(f"Battery/PEMFC power split per cluster (fixed seed): {POWER_SPLIT_PERCENT}%\n")
    for cluster in range(1, 11):
        problem.set_val(
            f"data:propulsion:he_power_train:DC_splitter:dc_splitter_{cluster}:power_split",
            units="percent",
            val=POWER_SPLIT_PERCENT,
        )

    try:
        problem.run_model()
    except AnalysisError as e:
        print(f"\n[FAILED] {e}\n")
        _print_nan_outputs(problem)
        raise
    except KeyboardInterrupt:
        # Ctrl+C during run_model() would normally skip straight past write_outputs() below --
        # write the current in-memory state anyway, to a separate "_interrupted" file so it's
        # never confused with a run that actually finished.
        print(
            "\n[INTERRUPTED] Ctrl+C received -- attempting to write output with current state.\n"
            "If the interruption landed mid Gauss-Seidel iteration (not right after a complete "
            "'NLBGS N' log line), this state may be slightly inconsistent between disciplines -- "
            "treat it as an approximate snapshot, not a final result.\n"
        )
        interrupted_path = problem.output_file_path.replace(".xml", "_interrupted.xml")
        problem.output_file_path = interrupted_path
        try:
            problem.write_outputs()
            print(f"Partial output written to {interrupted_path}\n")
            print("Summary (partial, at time of interruption):")
            print(f"  MTOW: {problem.get_val('data:weight:aircraft:MTOW', units='kg')} kg")
            print(
                f"  Total powertrain mass: "
                f"{problem.get_val('data:propulsion:he_power_train:mass', units='kg')} kg"
            )
        except Exception as write_error:
            print(f"Could not even write the partial output: {write_error}")
            print(
                "The Ctrl+C probably landed mid internal-write -- wait a moment and try "
                "interrupting again right after an 'NLBGS N' line."
            )
            raise
        return

    problem.write_outputs()
    print(f"\nRan without error. Full output written to {problem.output_file_path}\n")

    print("Summary:")
    print(f"  Converged MTOW: {problem.get_val('data:weight:aircraft:MTOW', units='kg')} kg")
    print(
        f"  Total powertrain mass: "
        f"{problem.get_val('data:propulsion:he_power_train:mass', units='kg')} kg"
    )
    print("\nNext step: python make_report.py  (builds figures + the .md/.html report)")


def main():
    """Opens a fresh log file per run (logs/<case>_<timestamp>.log, created next to this
    script) and mirrors everything that goes to the terminal -- including OpenMDAO's own
    prints (residuals, stall, convergence failure) -- to that file too, no manual redirection
    needed."""

    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"{CASE_NAME}_{timestamp}.log")

    stdout_orig, stderr_orig = sys.stdout, sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = _Tee(stdout_orig, log_file)
        sys.stderr = _Tee(stderr_orig, log_file)
        try:
            _run(log_path)
        finally:
            sys.stdout, sys.stderr = stdout_orig, stderr_orig

    print(f"\nFull log saved to: {log_path}")


if __name__ == "__main__":
    main()
