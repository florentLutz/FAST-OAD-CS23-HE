# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import pathlib
from shutil import rmtree
import logging

import pytest

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_better_cells"
ORIG_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
DOE_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_doe_better_cells"
DOE_RESULTS_FOLDER_PATH_SPLIT = (
    pathlib.Path(__file__).parent / "results_doe_power_split_better_cells"
)
WORKDIR_FOLDER_PATH = pathlib.Path(__file__).parent / "workdir"


@pytest.fixture(scope="module")
def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree(WORKDIR_FOLDER_PATH, ignore_errors=True)


def test_doe_sr22_hybrid_power_split():
    """
    Tests a hybrid sr22 with the same climb, cruise, descent and reserve profile as the original
    one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights) for various power
    split. The initial testing on this configuration revealed that the battery cells were sized in
    energy so we will use a high energy density cell in the Ampirius catalog.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid.yml"

    # power_splits = np.arange(60, 79.5, 1)
    power_splits = [80]

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Model options are set up straight into the configuration file
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # In addition to a change in battery production process, the characteristics of the battery
    # packs are changed. We will assume same polarization curve and small relative capacity effect.
    # We need at least 4 "fake" points for the relative capacity effect as it assumed to be a
    # deg 3 polynomial.
    problem.model_options["*"] = {
        "cell_capacity_ref": 4.000,
        "cell_weight_ref": 48.0e-3,
        "reference_curve_current": [100.0, 4000.0, 8000.0, 12000.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }
    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}

    problem.setup()

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=3.0,
        units="h**-1",
    )
    problem.set_val(
        "data:propulsion:he_power_train:PMSM:motor_1:voltage_caliber",
        val=830.0,
        units="V",
    )
    problem.set_val(
        "data:propulsion:he_power_train:PMSM:motor_1:rpm_rating",
        val=4060.0,
        units="min**-1",
    )

    for power_split in power_splits:
        problem.set_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
            power_split,
            units="percent",
        )
        problem.run_model()

        problem.output_file_path = DOE_RESULTS_FOLDER_PATH_SPLIT / (
            str(int(power_split)) + "_percent_split_mda.xml"
        )
        problem.write_outputs()


def test_optimization_sr22_hybrid():
    """
    Optimizes the hybrid sr22 with the same climb, cruise, descent and reserve profile as the o
    riginal one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights).
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.model.approx_totals()

    problem.model.add_design_var(
        name="data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        units="percent",
        lower=69.0,
        upper=85.0,
    )
    # We can't go lower than 69.0% otherwise we start requiring electric motor for which the rated
    # power is above what our model can handle.
    problem.model.add_objective(name="data:environmental_impact:sizing:emissions", units="kg")

    recorder = om.SqliteRecorder("driver_cases_better_cells.sql")
    problem.driver.add_recorder(recorder)

    problem.driver.recording_options["record_desvars"] = True
    problem.driver.recording_options["record_objectives"] = True

    # In addition to a change in battery production process, the characteristics of the battery
    # packs are changed. We will assume same polarization curve and small relative capacity effect.
    # We need at least 4 "fake" points for the relative capacity effect as it assumed to be a
    # deg 3 polynomial.
    problem.model_options["*"] = {
        "cell_capacity_ref": 4.000,
        "cell_weight_ref": 48.0e-3,
        "reference_curve_current": [100.0, 4000.0, 8000.0, 12000.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }

    problem.setup()

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=3.0,
        units="h**-1",
    )
    problem.set_val(
        "data:propulsion:he_power_train:PMSM:motor_1:voltage_caliber",
        val=830.0,
        units="V",
    )
    problem.set_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        val=80.0,
        units="percent",
    )

    problem.run_driver()
    problem.output_file_path = RESULTS_FOLDER_PATH / "optim_power_split.xml"
    problem.write_outputs()


def test_assess_op_perf_optimized_cirrus_sr22():
    """
    Looks at the performances on an op missions of the optimized Cirrus SR22
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    process_file_name = "op_mission_hybrid.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / "optim_power_split.xml"
    rename_variables_for_payload_range(ref_inputs)

    # Model options are set up straight into the configuration file
    problem.write_needed_inputs(ref_inputs)

    datafile = oad.DataFile(problem.input_file_path)
    datafile[
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split"
    ].value = np.full(92, 70.9275)
    datafile.save()

    problem.read_inputs()

    # In addition to a change in battery production process, the characteristics of the battery
    # packs are changed. We will assume same polarization curve and small relative capacity effect.
    # We need at least 4 "fake" points for the relative capacity effect as it assumed to be a
    # deg 3 polynomial.
    problem.model_options["*"] = {
        "cell_capacity_ref": 4.000,
        "cell_weight_ref": 48.0e-3,
        "reference_curve_current": [100.0, 4000.0, 8000.0, 12000.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }
    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": False}

    problem.setup()

    problem.set_val("data:mission:operational:range", val=100, units="NM")
    # We won't go below 100nm because at 100nm the electric Cirrus with better cell is capable of
    # doing this with much less emissions !

    # At 125, we start not being able to lower the hybridization ratio in cruise anymore because we
    # reach the constraints on the max shaft power, so we decrease the hybridzation ratio in other
    # phases of the flight: descent, taxi, ...
    power_split_cruise = 48
    power_split_descent = 0
    power_split_reserve = 65.5
    power_split_taxi = 0

    power_split = np.concatenate(
        (
            np.full(1, power_split_taxi),
            np.full(30, 70.9275),
            np.full(30, power_split_cruise),
            np.full(20, power_split_descent),
            np.full(10, power_split_reserve),
            np.full(1, power_split_taxi),
        )
    )

    problem.set_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        power_split,
        units="percent",
    )
    # The power split in *cruise* will be minimized to further decrease the fuel consumption in
    # *cruise* while we ensure SOC min is still 0 and c-rate is still below 1 (max for the cell).
    # Same for the e-motor shaft power

    problem.run_model()

    problem.write_outputs()

    print(
        problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
        )
    )
    print(
        problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:c_rate_max", units="h**-1"
        )[0]
        - 1.0
    )
    print(
        problem.get_val("data:propulsion:he_power_train:PMSM:motor_1:shaft_power_max", units="kW")[
            0
        ]
        - 91.11
    )


def test_assess_op_perf_optimized_cirrus_sr22_without_hybridization_ratio_change():
    """
    Looks at the performances on an op missions of the optimized Cirrus SR22 without adjusting the
    hybridization ratio
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    process_file_name = "op_mission_hybrid.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / "optim_power_split.xml"
    rename_variables_for_payload_range(ref_inputs)

    # Model options are set up straight into the configuration file
    problem.write_needed_inputs(ref_inputs)

    datafile = oad.DataFile(problem.input_file_path)
    datafile[
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split"
    ].value = np.full(92, 70.9275)
    datafile.save()

    problem.read_inputs()

    # In addition to a change in battery production process, the characteristics of the battery
    # packs are changed. We will assume same polarization curve and small relative capacity effect.
    # We need at least 4 "fake" points for the relative capacity effect as it assumed to be a
    # deg 3 polynomial.
    problem.model_options["*"] = {
        "cell_capacity_ref": 4.000,
        "cell_weight_ref": 48.0e-3,
        "reference_curve_current": [100.0, 4000.0, 8000.0, 12000.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }
    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": False}

    problem.setup()

    problem.set_val("data:mission:operational:range", val=100, units="NM")
    # We won't go below 100nm because at 100nm the electric Cirrus with better cell is capable of
    # doing this with much less emissions !

    problem.run_model()

    problem.write_outputs()


def test_sizing_sr22_hybrid_no_lto_improved():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()
    problem.input_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_in_no_LTO_improved.xml"

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)

    # Starting from the repartition in the previous test, we add a smoothing zone
    smoothing_zone_width = 10
    smoothing_zone_width_descent = min(smoothing_zone_width, 12)  # To avoid going into cruise
    smoothing_zone_width_climb = min(smoothing_zone_width, 22)  # To avoid going into cruise
    power_split = (
        100.0
        * np.concatenate(
            (
                np.zeros(12),  # Taxi out and climb
                np.linspace(0, 1, smoothing_zone_width_climb + 2)[1:-1],  # Transition
                np.ones(
                    61 - smoothing_zone_width_descent - smoothing_zone_width_climb
                ),  # End of climb, cruise and start of descent
                np.linspace(1, 0, smoothing_zone_width_descent + 2)[1:-1],  # Transition
                np.zeros(8),  # End of descent
                np.ones(10),  # Reserve
                np.zeros(1),  # Taxi in
            )
        )
    )
    datafile = oad.DataFile(problem.input_file_path)
    datafile[
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split"
    ].value = power_split
    datafile.save()

    problem.read_inputs()

    # We use here the 10 Ah Ultra-High Power Cell from Ampirius but using its original
    # characteristics we get a convergence error because one cell weighs too much. So instead, we
    # will use more smaller cells
    problem.model_options["*"] = {
        "cell_capacity_ref": 1.0000,
        "cell_weight_ref": 9.20e-3,
        "reference_curve_current": [10.0, 3300.0, 6600.0, 10000.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }
    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": False}

    problem.setup()

    problem.model.nonlinear_solver.options["use_aitken"] = True
    problem.model.nonlinear_solver.options["aitken_max_factor"] = 0.8
    problem.model.nonlinear_solver.options["aitken_min_factor"] = 0.33
    problem.model.nonlinear_solver.options["aitken_initial_factor"] = 0.8
    problem.model.nonlinear_solver.options["maxiter"] = 30
    problem.model.nonlinear_solver.options["stall_limit"] = 5
    problem.model.nonlinear_solver.options["stall_tol"] = 3e-6

    # We will need the biggest motor we can get, the EMRAX348
    problem.set_val(
        "data:propulsion:he_power_train:PMSM:motor_1:torque_max", val=1000.0, units="N*m"
    )
    problem.set_val(
        "data:propulsion:he_power_train:PMSM:motor_1:rpm_rating", val=3250.0, units="min**-1"
    )
    problem.set_val(
        "data:propulsion:he_power_train:PMSM:motor_1:voltage_caliber", val=830.0, units="V"
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=10.0,
        units="h**-1",
    )

    problem.run_model()

    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_out_no_LTO_improved.xml"
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)
    # Also rename the .csv so they are not overwritten (because the conf file and pt watcher files
    # are shared).
    if pathlib.Path(RESULTS_FOLDER_PATH / "hybrid_propulsion_no_LTO_improved.csv").exists():
        pathlib.Path(RESULTS_FOLDER_PATH / "hybrid_propulsion_no_LTO_improved.csv").unlink()
    mission_file = pathlib.Path(ORIG_RESULTS_FOLDER_PATH / "hybrid_propulsion.csv")
    mission_file.rename(RESULTS_FOLDER_PATH / "hybrid_propulsion_no_LTO_improved.csv")

    if pathlib.Path(
        RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher_no_LTO_improved.csv"
    ).exists():
        pathlib.Path(
            RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher_no_LTO_improved.csv"
        ).unlink()
    pt_watcher_file = pathlib.Path(ORIG_RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher.csv")
    pt_watcher_file.rename(RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher_no_LTO_improved.csv")


def rename_variables_for_payload_range(source_file_path: pathlib.Path):
    """
    Small helper function because payload range needs data based on the operational mission while
    for the sizing, the sizing mission is used.
    """

    op_name_to_sizing_name = {
        "data:mission:operational:climb:climb_rate:cruise_level": "data:mission:sizing:main_route:climb:climb_rate:cruise_level",
        "data:mission:operational:climb:climb_rate:sea_level": "data:mission:sizing:main_route:climb:climb_rate:sea_level",
        "data:mission:operational:cruise:altitude": "data:mission:sizing:main_route:cruise:altitude",
        "data:mission:operational:cruise:v_tas": "data:TLAR:v_cruise",
        "data:mission:operational:descent:descent_rate": "data:mission:sizing:main_route:descent:descent_rate",
        "data:mission:operational:initial_climb:energy": "data:mission:sizing:initial_climb:energy",
        "data:mission:operational:initial_climb:fuel": "data:mission:sizing:initial_climb:fuel",
        "data:mission:operational:payload:CG:x": "data:weight:payload:PAX:CG:x",
        "data:mission:operational:payload:mass": "data:weight:aircraft:payload",
        "data:mission:operational:range": "data:TLAR:range",
        "data:mission:operational:reserve:altitude": "data:mission:sizing:main_route:reserve:altitude",
        "data:mission:operational:reserve:duration": "data:mission:sizing:main_route:reserve:duration",
        "data:mission:operational:takeoff:energy": "data:mission:sizing:takeoff:energy",
        "data:mission:operational:takeoff:fuel": "data:mission:sizing:takeoff:fuel",
        "data:mission:operational:taxi_in:duration": "data:mission:sizing:taxi_in:duration",
        "data:mission:operational:taxi_in:speed": "data:mission:sizing:taxi_in:speed",
        "data:mission:operational:taxi_out:duration": "data:mission:sizing:taxi_out:duration",
        "data:mission:operational:taxi_out:speed": "data:mission:sizing:taxi_out:speed",
        "data:mission:operational:reserve:v_tas": "data:mission:sizing:main_route:reserve:v_tas",
    }

    datafile = oad.DataFile(source_file_path)

    for op_name, sizing_name in op_name_to_sizing_name.items():
        variable_to_add = oad.Variable(
            op_name, val=datafile[sizing_name].value, units=datafile[sizing_name].units
        )
        datafile.append(variable_to_add)

    datafile.save()
