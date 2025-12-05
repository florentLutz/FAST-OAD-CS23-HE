# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
import pathlib
import time
import warnings

import openmdao.api as om
import numpy as np
import pandas as pd


from pyDOE2 import lhs

from .turboshaft_components.turboshaft_geometry_computation import DesignPointCalculation
from .turboshaft_components.turboshaft_off_design_fuel import Turboshaft


def get_ivc_all_data(power_design, t41t_design, opr_design, altitude_design, mach_design):
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

    return ivc


def run_design_point(ivc):
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

    fuel_consumed_design = prob.get_val("fuel_mass_flow", units="kg/h")[0]

    _, _, residuals = prob.model.get_nonlinear_vectors()
    norm = np.linalg.norm(residuals.asarray())

    if prob.model.nonlinear_solver._iter_count < 100 and not np.isnan(norm) and not np.isinf(norm):
        converged = [True]
    else:
        converged = [False]

    return alpha, alpha_p, a41, a45, a8, opr_2_opr_1, fuel_consumed_design, converged


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


def run_off_design_fuel(
    shaft_power_off_design: list,
    altitude_off_design: list,
    mach_off_design: list,
    ivc: om.IndepVarComp,
):
    fuel_consumed = []
    exhaust_mass_flow = []
    exhaust_velocity = []
    p3t = []
    t3t = []
    converged = []

    ivc.add_output("altitude", val=altitude_off_design, units="ft")
    ivc.add_output("mach_0", val=mach_off_design)
    ivc.add_output("required_shaft_power", val=shaft_power_off_design, units="kW")

    prob = get_fuel_problem(ivc)

    prob.setup()

    prob.run_model()

    _, _, residuals = prob.model.get_nonlinear_vectors()
    norm = np.linalg.norm(residuals.asarray())

    fuel_consumed.append(prob.get_val("fuel_mass_flow", units="kg/h")[0])
    exhaust_mass_flow.append(
        prob.get_val("air_mass_flow", units="kg/s")[0]
        * (
            1.0
            + prob.get_val("fuel_air_ratio")[0]
            - prob.get_val("compressor_bleed_ratio")[0]
            - prob.get_val("pressurization_bleed_ratio")[0]
        )
    )
    exhaust_velocity.append(prob.get_val("velocity_8", units="m/s")[0])
    t3t.append(prob.get_val("total_temperature_3", units="degK")[0])
    p3t.append(prob.get_val("total_pressure_3", units="bar")[0])

    if prob.model.nonlinear_solver._iter_count < 100 and not np.isnan(norm) and not np.isinf(norm):
        converged.append(True)
    else:
        converged.append(False)
        prob = get_fuel_problem(ivc)
        prob.setup()

    # i = 1

    # for altitude, mach, power in zip(
    #     altitude_off_design[1:], mach_off_design[1:], shaft_power_off_design[1:]
    # ):
    #     prob.set_val("altitude", val=altitude, units="ft")
    #     prob.set_val("mach_0", val=mach)
    #     prob.set_val("required_shaft_power", val=power, units="kW")
    #
    #     if i % 10 == 0:
    #         print(str(i / 5) + " %")
    #
    #     prob.run_model()
    #     _, _, residuals = prob.model.get_nonlinear_vectors()
    #     norm = np.linalg.norm(residuals.asarray())
    #
    #     fuel_consumed.append(prob.get_val("fuel_mass_flow", units="kg/h")[0])
    #     exhaust_thrust.append(prob.get_val("exhaust_thrust", units="N")[0])
    #     t3t.append(prob.get_val("total_temperature_3", units="degK")[0])
    #     p3t.append(prob.get_val("total_pressure_3", units="bar")[0])
    #
    #     if (
    #         prob.model.nonlinear_solver._iter_count < 100
    #         and not np.isnan(norm)
    #         and not np.isinf(norm)
    #     ):
    #         converged.append(True)
    #     else:
    #         converged.append(False)
    #         prob = get_fuel_problem(ivc)
    #         prob.setup()
    #
    #         print(
    #             prob.model.nonlinear_solver._iter_count < 100
    #             and not np.isnan(norm)
    #             and not np.isinf(norm)
    #         )
    #
    #     i += 1

    return converged, fuel_consumed, exhaust_mass_flow, exhaust_velocity, t3t, p3t


def run_and_save_turboshaft_full_performances(
    power_design,
    t41t_design,
    opr_design,
    shaft_power_limit,
    altitude,
    mach,
    power_rate,
    data_folder_pth,
    save=True,
):
    print("Design thermodynamic power: " + str(power_design) + " kW")
    print("Design TET: " + str(t41t_design) + " degK")
    print("Design OPR: " + str(opr_design))
    print("Design altitude: " + str(0.0) + " ft")
    print("Design Mach number: " + str(0.0))
    print("Limit rated power: " + str(shaft_power_limit) + " kW")
    print("Altitude (ft):", altitude)
    print("Mach (-):", mach)
    print("Power rate (-):", power_rate)

    data_ivc = get_ivc_all_data(power_design, t41t_design, opr_design, 0.0, 0.0)

    (
        alpha,
        alpha_p,
        a41,
        a45,
        a8,
        opr_2_opr_1,
        fuel_consumed_design,
        design_converged,
    ) = run_design_point(data_ivc)

    data_ivc.add_output("data:propulsion:turboprop:section:41", val=a41, units="m**2")
    data_ivc.add_output("data:propulsion:turboprop:section:45", val=a45, units="m**2")
    data_ivc.add_output("data:propulsion:turboprop:section:8", val=a8, units="m**2")
    data_ivc.add_output("data:propulsion:turboprop:design_point:alpha", val=alpha)
    data_ivc.add_output("data:propulsion:turboprop:design_point:alpha_p", val=alpha_p)
    data_ivc.add_output("data:propulsion:turboprop:design_point:opr_2_opr_1", val=opr_2_opr_1)

    # fig = go.Figure()
    #
    # for idx, alt in enumerate(altitude_list):
    #     scatter_current_alt_opr_limit = go.Scatter(
    #         x=mach_list,
    #         y=max_power_opr[idx :: len(altitude_list)],
    #         mode="lines+markers",
    #         name="Max power OPR limit altitude" + str(alt),
    #         legendgroup=str(alt),
    #         legendgrouptitle_text=str(alt),
    #     )
    #     fig.add_trace(scatter_current_alt_opr_limit)
    #     scatter_current_alt_itt_limit = go.Scatter(
    #         x=mach_list,
    #         y=max_power_itt[idx :: len(altitude_list)],
    #         mode="lines+markers",
    #         name="Max power ITT limit altitude" + str(alt),
    #         legendgroup=str(alt),
    #     )
    #     fig.add_trace(scatter_current_alt_itt_limit)
    #
    # fig.show()

    if design_converged[0]:
        (
            converged,
            fuel_consumed,
            exhaust_mass_flow,
            exhaust_velocity,
            t3t,
            p3t,
        ) = run_off_design_fuel(power_rate * shaft_power_limit, altitude, mach, data_ivc)
    else:
        _, fuel_consumed, exhaust_mass_flow, exhaust_velocity, t3t, p3t = run_off_design_fuel(
            power_rate * shaft_power_limit, altitude, mach, data_ivc
        )
        converged = [False]

    converged = np.array(converged)

    altitude_for_save_save = np.array([altitude])[converged]
    mach_for_save_save = np.array([mach])[converged]
    power_for_save_save = np.array([power_rate * shaft_power_limit])[converged]
    design_power_for_save_save = np.full_like(altitude_for_save_save, power_design)
    design_t41t_for_save_save = np.full_like(altitude_for_save_save, t41t_design)
    design_opr_for_save_save = np.full_like(altitude_for_save_save, opr_design)
    design_fuel_consumed_for_save_save = np.full_like(altitude_for_save_save, fuel_consumed_design)
    design_altitude_for_save_save = np.full_like(altitude_for_save_save, 0.0)
    design_mach_for_save_save = np.full_like(altitude_for_save_save, 0.0)
    shaft_limit_for_save_save = np.full_like(altitude_for_save_save, shaft_power_limit)
    fuel_consumed_for_save_save = np.array(fuel_consumed)[converged]
    exhaust_mass_flow_for_save_save = np.array(exhaust_mass_flow)[converged]
    exhaust_velocity_for_save_save = np.array(exhaust_velocity)[converged]
    t3t_thrust_for_save_save = np.array(t3t)[converged]
    p3t_thrust_for_save_save = np.array(p3t)[converged]

    data_fc = np.c_[
        design_power_for_save_save,
        design_t41t_for_save_save,
        design_opr_for_save_save,
        design_altitude_for_save_save,
        design_mach_for_save_save,
        design_fuel_consumed_for_save_save,
        shaft_limit_for_save_save,
        altitude_for_save_save,
        mach_for_save_save,
        power_for_save_save,
        fuel_consumed_for_save_save,
        exhaust_mass_flow_for_save_save,
        exhaust_velocity_for_save_save,
        t3t_thrust_for_save_save,
        p3t_thrust_for_save_save,
    ]

    result_dataframe_fc = pd.DataFrame(
        data_fc,
        columns=[
            "Design Power (kW)",
            "Design T41t (degK)",
            "Design OPR (-)",
            "Design altitude (ft)",
            "Design Mach (-)",
            "Design Fuel Consumed (kg/h)",
            "Limit Shaft (kW)",
            "Altitude (ft)",
            "Mach (-)",
            "Shaft Power (kW)",
            "Fuel mass flow (kg/h)",
            "Exhaust mass flow (kg/s)",
            "Exhaust velocity (m/s)",
            "Total temperature 3 (degK)",
            "Total pressure 3 (bar)",
        ],
    )

    result_file_path_fc = data_folder_pth / "fuel_consumed_v2.csv"

    if save:
        if result_file_path_fc.exists():
            existing_data = pd.read_csv(result_file_path_fc, index_col=0)
            stacked_dataframe = pd.concat([existing_data, result_dataframe_fc], axis=0)
            stacked_dataframe.to_csv(result_file_path_fc)

        else:
            result_dataframe_fc.to_csv(result_file_path_fc)
    else:
        print(altitude, mach, power_rate * shaft_power_limit, fuel_consumed)


if __name__ == "__main__":
    path_to_current_file = pathlib.Path(__file__)
    parent_folder = path_to_current_file.parents[0]
    data_folder_path = parent_folder / "data"

    doeX = lhs(9, samples=10, criterion="correlation")

    power_min_kW, power_max_kW = (190.0, 354.0)
    power_ratio_min, power_ratio_max = (1.0, 2.5)
    opr_power_ratio_min, opr_power_ratio_max = (0.01, 0.017)
    opr_ratio_min, opr_ratio_max = (1.0, 1.3)
    itt_power_ratio_min, itt_power_ratio_max = (
        0.66,
        2.0,
    )
    temperature_ratio_min, temperature_ratio_max = (0.7, 0.85)
    altitude_min, altitude_max = (0.0, 35000.0)
    mach_min, mach_max = (0.0, 0.6)
    power_rate_min, power_rate_max = (0.4, 1.0)

    doeX[:, 0] = doeX[:, 0] * (1.05 * power_max_kW - 0.95 * power_min_kW) + power_min_kW
    doeX[:, 1] = (doeX[:, 1] * (power_ratio_max - power_ratio_min) + power_ratio_min) * doeX[:, 0]
    doeX[:, 2] = np.maximum(
        (doeX[:, 2] * (opr_power_ratio_max - opr_power_ratio_min) + opr_power_ratio_min)
        * doeX[:, 0],
        np.full_like(doeX[:, 0], 5),
    )
    doeX[:, 3] = (doeX[:, 3] * (opr_ratio_max - opr_ratio_min) + opr_ratio_min) * doeX[:, 2]
    doeX[:, 4] = np.clip(
        (doeX[:, 4] * (itt_power_ratio_max - itt_power_ratio_min) + itt_power_ratio_min)
        * doeX[:, 0],
        np.full_like(doeX[:, 2], 600),
        np.full_like(doeX[:, 2], 1100),
    )
    doeX[:, 5] = (
        doeX[:, 5] * (temperature_ratio_max - temperature_ratio_min) + power_ratio_min
    ) * doeX[:, 4]
    doeX[:, 6] = doeX[:, 6] * (altitude_max - altitude_min) + altitude_min
    doeX[:, 7] = doeX[:, 7] * (mach_max - mach_min) + mach_min
    doeX[:, 8] = doeX[:, 8] * (power_rate_max - power_rate_min) + power_rate_min

    for turboshaft_design_parameter in doeX:
        t1 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_and_save_turboshaft_full_performances(
                turboshaft_design_parameter[1],
                turboshaft_design_parameter[5] + 273.15,
                turboshaft_design_parameter[2],
                turboshaft_design_parameter[0],
                turboshaft_design_parameter[6],
                turboshaft_design_parameter[7],
                turboshaft_design_parameter[8],
                data_folder_path,
            )
        t2 = time.time()
        print("Turboprop done after " + str(t2 - t1) + " s!")
