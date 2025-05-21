# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
import copy
import pathlib

import openmdao.api as om
import numpy as np
import pandas as pd
import pytest

from turboshaft_components.turboshaft_geometry_computation import DesignPointCalculation
from turboshaft_components.turboshaft_off_design_fuel import Turboshaft

THERMODYNAMIC_POWER_COLUMN_NAME = "Design Power (kW)"
OPR_COLUMN_NAME = "Design OPR (-)"
T41T_COLUMN_NAME = "Design T41t (degK)"
OPR_LIMIT_COLUMN_NAME = "Limit OPR (-)"
ITT_LIMIT_COLUMN_NAME = "Limit ITT (degK)"


def identify_design(df: pd.DataFrame):
    """
    We'll assume that the chance of two designs in the data having the same design power is
    minimal (though not theoretically impossible) so we'll identify designs by their power.
    """

    design_powers = df[THERMODYNAMIC_POWER_COLUMN_NAME].to_list()
    unique_design_powers = list(set(design_powers))

    design_opr = []
    design_tet = []
    opr_limit = []
    itt_limit = []

    for design_power in unique_design_powers:
        current_design_df = df.loc[df[THERMODYNAMIC_POWER_COLUMN_NAME] == design_power]
        design_opr.append(current_design_df[OPR_COLUMN_NAME].to_numpy()[0])
        design_tet.append(current_design_df[T41T_COLUMN_NAME].to_numpy()[0])
        opr_limit.append(current_design_df[OPR_LIMIT_COLUMN_NAME].to_numpy()[0])
        itt_limit.append(current_design_df[ITT_LIMIT_COLUMN_NAME].to_numpy()[0])

    return unique_design_powers, design_opr, design_tet, opr_limit, itt_limit


def get_ivc_all_data(
    power_design, t41t_design, opr_design, altitude_design, mach_design, opr_limit, itt_limit
):
    ivc = om.IndepVarComp()
    ivc.add_output("compressor_bleed_mass_flow", val=0.04, units="kg/s")
    ivc.add_output("cooling_bleed_ratio", val=0.025)
    ivc.add_output("pressurization_bleed_ratio", val=0.05)

    ivc.add_output("eta_225", val=0.85)
    ivc.add_output("eta_253", val=0.86)
    ivc.add_output("eta_445", val=0.86)
    ivc.add_output("eta_455", val=0.86)
    ivc.add_output("total_pressure_loss_02", val=0.8)
    ivc.add_output("pressure_loss_34", val=0.95)
    ivc.add_output("combustion_energy", val=43.260e6 * 0.95, units="J/kg")

    ivc.add_output("electric_power", val=0.0, units="hp")

    ivc.add_output(
        "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio",
        val=0.25,
    )
    ivc.add_output(
        "settings:propulsion:turboprop:efficiency:high_pressure_axe",
        val=0.98,
    )
    ivc.add_output(
        "settings:propulsion:turboprop:efficiency:gearbox",
        val=0.98,
    )
    ivc.add_output(
        "settings:propulsion:turboprop:design_point:mach_exhaust",
        val=0.4,
    )

    ivc.add_output(
        "data:propulsion:turboprop:design_point:altitude",
        val=altitude_design,
        units="m",
    )
    ivc.add_output("data:propulsion:turboprop:design_point:mach", val=mach_design)
    ivc.add_output(
        "data:propulsion:turboprop:design_point:power",
        val=power_design,
        units="kW",
    )
    ivc.add_output(
        "data:propulsion:turboprop:design_point:turbine_entry_temperature",
        val=t41t_design,
        units="degK",
    )
    ivc.add_output(
        "data:propulsion:turboprop:design_point:OPR",
        val=opr_design,
    )
    ivc.add_output("itt_limit", val=itt_limit, units="degK")
    ivc.add_output("opr_limit", val=opr_limit)

    return ivc


def get_fuel_problem(ivc):
    prob = om.Problem(reports=False)
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem(
        "turboshaft_off_design_fuel",
        Turboshaft(number_of_points=1),
        promotes=["*"],
    )

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
    prob.model.nonlinear_solver.linesearch.options["maxiter"] = 5
    prob.model.nonlinear_solver.linesearch.options["alpha"] = 1.5
    prob.model.nonlinear_solver.linesearch.options["c"] = 2e-1
    prob.model.nonlinear_solver.options["iprint"] = 0
    prob.model.nonlinear_solver.options["maxiter"] = 100
    prob.model.nonlinear_solver.options["rtol"] = 1e-5
    prob.model.nonlinear_solver.options["atol"] = 1e-5
    prob.model.linear_solver = om.DirectSolver()

    return prob


def get_missing_turboshaft_data(ivc, altitudes, machs, shaft_powers):
    # First get the turboshaft geometry parameters from design
    prob = om.Problem(reports=False)
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem(
        "turboshaft_sizing",
        DesignPointCalculation(number_of_points=1),
        promotes=["*"],
    )

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
    prob.model.nonlinear_solver.options["iprint"] = 0
    prob.model.nonlinear_solver.options["maxiter"] = 100
    prob.model.nonlinear_solver.options["rtol"] = 1e-5
    prob.model.nonlinear_solver.options["atol"] = 5e-5
    prob.model.linear_solver = om.DirectSolver()

    prob.setup()

    prob.run_model()

    alpha = prob.get_val("data:propulsion:turboprop:design_point:alpha")[0]
    alpha_p = prob.get_val("data:propulsion:turboprop:design_point:alpha_p")[0]
    a41 = prob.get_val("data:propulsion:turboprop:section:41", units="m**2")[0]
    a45 = prob.get_val("data:propulsion:turboprop:section:45", units="m**2")[0]
    a8 = prob.get_val("data:propulsion:turboprop:section:8", units="m**2")[0]
    opr_2_opr_1 = prob.get_val("opr_2")[0] / prob.get_val("opr_1")[0]

    ivc.add_output("data:propulsion:turboprop:section:41", val=a41, units="m**2")
    ivc.add_output("data:propulsion:turboprop:section:45", val=a45, units="m**2")
    ivc.add_output("data:propulsion:turboprop:section:8", val=a8, units="m**2")
    ivc.add_output("data:propulsion:turboprop:design_point:alpha", val=alpha)
    ivc.add_output("data:propulsion:turboprop:design_point:alpha_p", val=alpha_p)
    ivc.add_output("data:propulsion:turboprop:design_point:opr_2_opr_1", val=opr_2_opr_1)

    exhaust_velocity = []
    exhaust_mass_flow = []
    fuel_mass_flow = []
    converged = []

    ivc.add_output("altitude", val=altitudes[0], units="ft")
    ivc.add_output("mach_0", val=machs[0])
    ivc.add_output("required_shaft_power", val=shaft_powers[0], units="kW")

    prob = get_fuel_problem(ivc)

    prob.setup()

    prob.run_model()

    _, _, residuals = prob.model.get_nonlinear_vectors()
    norm = np.linalg.norm(residuals.asarray())

    exhaust_velocity.append(prob.get_val("velocity_8", units="m/s")[0])
    exhaust_mass_flow.append(
        prob.get_val("air_mass_flow", units="kg/s")[0]
        * (
            1.0
            + prob.get_val("fuel_air_ratio")[0]
            - prob.get_val("compressor_bleed_ratio")[0]
            - prob.get_val("pressurization_bleed_ratio")[0]
        )
    )
    fuel_mass_flow.append(prob.get_val("fuel_mass_flow", units="kg/h")[0])

    if prob.model.nonlinear_solver._iter_count < 100 and not np.isnan(norm) and not np.isinf(norm):
        converged.append(True)
    else:
        converged.append(False)
        prob = get_fuel_problem(ivc)
        prob.setup()

    i = 1

    for altitude, mach, shaft_power in zip(altitudes[1:], machs[1:], shaft_powers[1:]):
        prob.set_val("altitude", val=altitude, units="ft")
        prob.set_val("mach_0", val=mach)
        prob.set_val("required_shaft_power", val=shaft_power, units="kW")

        if i % 10 == 0:
            print(str(i / 5) + " %")

        prob.run_model()
        _, _, residuals = prob.model.get_nonlinear_vectors()
        norm = np.linalg.norm(residuals.asarray())

        exhaust_velocity.append(prob.get_val("velocity_8", units="m/s")[0])
        exhaust_mass_flow.append(
            prob.get_val("air_mass_flow", units="kg/s")[0]
            * (
                1.0
                + prob.get_val("fuel_air_ratio")[0]
                - prob.get_val("compressor_bleed_ratio")[0]
                - prob.get_val("pressurization_bleed_ratio")[0]
            )
        )
        fuel_mass_flow.append(prob.get_val("fuel_mass_flow", units="kg/h")[0])

        if (
            prob.model.nonlinear_solver._iter_count < 100
            and not np.isnan(norm)
            and not np.isinf(norm)
        ):
            converged.append(True)
        else:
            converged.append(False)
            prob = get_fuel_problem(ivc)
            prob.setup()

        i += 1

    return exhaust_mass_flow, exhaust_velocity, converged, fuel_mass_flow


if __name__ == "__main__":
    path_to_current_file = pathlib.Path(__file__)
    parent_folder = path_to_current_file.parents[0]
    data_folder_path = parent_folder / "data"

    path_to_fuel_consumed_file = data_folder_path / "fuel_consumed.csv"

    existing_data = pd.read_csv(path_to_fuel_consumed_file, index_col=0)

    new_data = copy.deepcopy(existing_data)
    new_data["Exhaust velocity (m/s)"] = np.nan
    new_data["Exhaust mass flow (kg/s)"] = np.nan

    design_power_list, design_oprs, design_t41ts, opr_limits, itt_limits = identify_design(
        existing_data
    )

    for index, power in enumerate(design_power_list):
        ivc_data = get_ivc_all_data(
            power,
            design_t41ts[index],
            design_oprs[index],
            0.0,
            0.0,
            opr_limits[index],
            itt_limits[index],
        )

        current_design_df = new_data.loc[new_data[THERMODYNAMIC_POWER_COLUMN_NAME] == power]

        altitude_to_evaluate = current_design_df["Altitude (ft)"]
        mach_to_evaluate = current_design_df["Mach (-)"]
        power_to_evaluate = current_design_df["Shaft Power (kW)"]

        (
            exhaust_mass_flows,
            exhaust_velocities,
            converged_idx,
            fuel_mass_flows,
        ) = get_missing_turboshaft_data(
            ivc_data, altitude_to_evaluate, mach_to_evaluate, power_to_evaluate
        )

        assert np.array(fuel_mass_flows)[np.array(converged_idx)] == pytest.approx(
            current_design_df["Fuel mass flow (kg/h)"].to_numpy(), rel=1e-2
        )

        new_data.loc[
            new_data[THERMODYNAMIC_POWER_COLUMN_NAME] == power, "Exhaust velocity (m/s)"
        ] = np.array(exhaust_velocities)[np.array(converged_idx)]
        new_data.loc[
            new_data[THERMODYNAMIC_POWER_COLUMN_NAME] == power, "Exhaust mass flow (kg/s)"
        ] = np.array(exhaust_mass_flows)[np.array(converged_idx)]

    result_file_path_fuel_consumed_complemented = (
        data_folder_path / "fuel_consumed_complemented.csv"
    )
    new_data.to_csv(result_file_path_fuel_consumed_complemented)
