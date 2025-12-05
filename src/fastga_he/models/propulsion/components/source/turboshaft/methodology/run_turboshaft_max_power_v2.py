# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import copy
import pathlib
import time
import warnings

import openmdao.api as om
import numpy as np
import pandas as pd

from pyDOE2 import lhs

from .turboshaft_components.turboshaft_geometry_computation import DesignPointCalculation
from .turboshaft_components.turboshaft_off_design_max_power import (
    TurboshaftMaxPowerOPRLimit,
    TurboshaftMaxPowerITTLimit,
)


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

    return alpha, alpha_p, a41, a45, a8, opr_2_opr_1, fuel_consumed_design


def run_max_power_opr_limit(altitude_off_design: list, mach_off_design: list, ivc: om.IndepVarComp):
    ivc.add_output("altitude", val=altitude_off_design, units="ft")
    ivc.add_output("mach_0", val=mach_off_design)

    prob = om.Problem(reports=False)
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem(
        "turboshaft_off_design_max_power_opr_limit",
        TurboshaftMaxPowerOPRLimit(number_of_points=1),
        promotes=["*"],
    )

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
    prob.model.nonlinear_solver.linesearch.options["maxiter"] = 5
    prob.model.nonlinear_solver.linesearch.options["alpha"] = 1.7
    prob.model.nonlinear_solver.linesearch.options["c"] = 2e-1
    prob.model.nonlinear_solver.options["iprint"] = 0
    prob.model.nonlinear_solver.options["maxiter"] = 100
    prob.model.nonlinear_solver.options["rtol"] = 1e-5
    prob.model.nonlinear_solver.options["atol"] = 5e-5
    prob.model.linear_solver = om.DirectSolver()

    prob.setup()

    prob.run_model()
    _, _, residuals = prob.model.get_nonlinear_vectors()
    norm = np.linalg.norm(residuals.asarray())

    max_power = prob.get_val("required_shaft_power", units="kW")[0]

    if prob.model.nonlinear_solver._iter_count < 100 and not np.isnan(norm) and not np.isinf(norm):
        return max_power, True
    else:
        return max_power, False


def run_max_power_itt_limit(altitude_off_design: list, mach_off_design: list, ivc: om.IndepVarComp):
    ivc.add_output("altitude", val=altitude_off_design, units="ft")
    ivc.add_output("mach_0", val=mach_off_design)

    prob = om.Problem(reports=False)
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem(
        "turboshaft_off_design_max_power_opr_limit",
        TurboshaftMaxPowerITTLimit(number_of_points=1),
        promotes=["*"],
    )

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
    prob.model.nonlinear_solver.linesearch.options["maxiter"] = 5
    prob.model.nonlinear_solver.linesearch.options["alpha"] = 1.7
    prob.model.nonlinear_solver.linesearch.options["c"] = 2e-1
    prob.model.nonlinear_solver.options["iprint"] = 0
    prob.model.nonlinear_solver.options["maxiter"] = 100
    prob.model.nonlinear_solver.options["rtol"] = 1e-5
    prob.model.nonlinear_solver.options["atol"] = 5e-5
    prob.model.linear_solver = om.DirectSolver()

    prob.setup()

    prob.run_model()
    _, _, residuals = prob.model.get_nonlinear_vectors()
    norm = np.linalg.norm(residuals.asarray())

    max_power = prob.get_val("required_shaft_power", units="kW")[0]

    if prob.model.nonlinear_solver._iter_count < 100 and not np.isnan(norm) and not np.isinf(norm):
        return max_power, True
    else:
        return max_power, False


def run_turboshaft_max_power(
    power_design,
    t41t_design,
    opr_design,
    opr_limit,
    itt_limit,
    altitude,
    mach,
):
    print("Design thermodynamic power: " + str(power_design) + " kW")
    print("Design TET: " + str(t41t_design) + " degK")
    print("Design OPR: " + str(opr_design))
    print("Design altitude: " + str(0.0) + " ft")
    print("Design Mach number: " + str(0.0))
    print("Limit OPR: " + str(opr_limit))
    print("Limit ITT: " + str(itt_limit) + " degK")

    data_ivc = get_ivc_all_data(
        power_design, t41t_design, opr_design, 0.0, 0.0, opr_limit, itt_limit
    )

    alpha, alpha_p, a41, a45, a8, opr_2_opr_1, fuel_consumed_design = run_design_point(data_ivc)

    data_ivc.add_output("data:propulsion:turboprop:section:41", val=a41, units="m**2")
    data_ivc.add_output("data:propulsion:turboprop:section:45", val=a45, units="m**2")
    data_ivc.add_output("data:propulsion:turboprop:section:8", val=a8, units="m**2")
    data_ivc.add_output("data:propulsion:turboprop:design_point:alpha", val=alpha)
    data_ivc.add_output("data:propulsion:turboprop:design_point:alpha_p", val=alpha_p)
    data_ivc.add_output("data:propulsion:turboprop:design_point:opr_2_opr_1", val=opr_2_opr_1)

    data_ivc_bis = copy.deepcopy(data_ivc)

    max_power_opr, converged_opr = run_max_power_opr_limit(altitude, mach, data_ivc)
    max_power_itt, converged_itt = run_max_power_itt_limit(altitude, mach, data_ivc_bis)

    converged_max_power = converged_opr & converged_itt

    return max_power_opr, max_power_itt, converged_max_power


if __name__ == "__main__":
    path_to_current_file = pathlib.Path(__file__)
    parent_folder = path_to_current_file.parents[0]
    data_folder_path = parent_folder / "data"

    doeX = lhs(8, samples=500, criterion="correlation")

    power_min_kW, power_max_kW = (354.0, 1268.0)
    power_ratio_min, power_ratio_max = (1.0, 2.5)
    opr_power_ratio_min, opr_power_ratio_max = (0.01, 0.017)
    opr_ratio_min, opr_ratio_max = (1.0, 1.3)
    itt_power_ratio_min, itt_power_ratio_max = (
        0.66,
        2.0,
    )  # Here we'll check that we don't get absurd values (>685 degC and <1000 degC)
    temperature_ratio_min, temperature_ratio_max = (0.7, 0.85)
    altitude_min, altitude_max = (0.0, 35000.0)
    mach_min, mach_max = (0.05, 0.6)

    doeX[:, 0] = doeX[:, 0] * (1.05 * power_max_kW - 0.95 * power_min_kW) + power_min_kW
    doeX[:, 1] = (doeX[:, 1] * (power_ratio_max - power_ratio_min) + power_ratio_min) * doeX[:, 0]
    doeX[:, 2] = np.maximum(
        (doeX[:, 2] * (opr_power_ratio_max - opr_power_ratio_min) + opr_power_ratio_min)
        * doeX[:, 0],
        np.full_like(doeX[:, 0], 7),
    )
    doeX[:, 3] = (doeX[:, 3] * (opr_ratio_max - opr_ratio_min) + opr_ratio_min) * doeX[:, 2]
    doeX[:, 4] = np.clip(
        (doeX[:, 4] * (itt_power_ratio_max - itt_power_ratio_min) + itt_power_ratio_min)
        * doeX[:, 0],
        np.full_like(doeX[:, 2], 685),
        np.full_like(doeX[:, 2], 1000),
    )
    doeX[:, 5] = (
        doeX[:, 5] * (temperature_ratio_max - temperature_ratio_min) + power_ratio_min
    ) * doeX[:, 4]
    doeX[:, 6] = doeX[:, 6] * (altitude_max - altitude_min) + altitude_min
    doeX[:, 7] = doeX[:, 7] * (mach_max - mach_min) + mach_min

    max_power_opr_list = []
    max_power_itt_list = []
    rated_power = []
    thermodynamic_power_design = []
    opr_design_list = []
    opr_limit_list = []
    itt_limit_list = []
    tet_design = []
    altitude_list = []
    mach_list = []

    for turboshaft_design_parameter in doeX:
        try:
            t1 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                max_power_opr, max_power_itt, converged = run_turboshaft_max_power(
                    turboshaft_design_parameter[1],
                    turboshaft_design_parameter[5] + 273.15,
                    turboshaft_design_parameter[2],
                    turboshaft_design_parameter[3],
                    turboshaft_design_parameter[4] + 273.15,
                    turboshaft_design_parameter[6],
                    turboshaft_design_parameter[7],
                )

                if converged:
                    max_power_opr_list.append(max_power_opr)
                    max_power_itt_list.append(max_power_itt)
                    rated_power.append(turboshaft_design_parameter[0])
                    thermodynamic_power_design.append(turboshaft_design_parameter[1])
                    opr_design_list.append(turboshaft_design_parameter[2])
                    opr_limit_list.append(turboshaft_design_parameter[3])
                    itt_limit_list.append(turboshaft_design_parameter[4] + 273.15)
                    tet_design.append(turboshaft_design_parameter[5] + 273.15)
                    altitude_list.append(turboshaft_design_parameter[6])
                    mach_list.append(turboshaft_design_parameter[7])

            t2 = time.time()

            print("Turboprop done after " + str(t2 - t1) + " s!")
        # Bad, I know
        except:  # noqa: E722
            pass

    data = np.c_[
        np.array(thermodynamic_power_design),
        np.array(tet_design),
        np.array(opr_design_list),
        np.zeros_like(np.array(thermodynamic_power_design)),
        np.zeros_like(np.array(thermodynamic_power_design)),
        np.array(opr_limit_list),
        np.array(itt_limit_list),
        np.array(altitude_list),
        np.array(mach_list),
        np.array(max_power_opr_list),
        np.array(max_power_itt_list),
    ]

    result_file_path_max_power = data_folder_path / "max_power_v2.csv"

    result_dataframe_max_power = pd.DataFrame(
        data,
        columns=[
            "Design Power (kW)",
            "Design T41t (degK)",
            "Design OPR (-)",
            "Design altitude (ft)",
            "Design Mach (-)",
            "Limit OPR (-)",
            "Limit ITT (degK)",
            "Altitude (ft)",
            "Mach (-)",
            "Max Power OPR Limit (kW)",
            "Max Power ITT Limit (kW)",
        ],
    )

    if result_file_path_max_power.exists():
        existing_data = pd.read_csv(result_file_path_max_power, index_col=0)
        stacked_dataframe = pd.concat([existing_data, result_dataframe_max_power], axis=0)
        stacked_dataframe.to_csv(result_file_path_max_power)

    else:
        result_dataframe_max_power.to_csv(result_file_path_max_power)
