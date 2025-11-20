# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import pathlib
import logging

import pytest
import numpy as np

import openmdao.api as om
import fastoad.api as oad

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
DOE_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_doe"
DOE_RESULTS_FOLDER_PATH_SPLIT = pathlib.Path(__file__).parent / "results_doe_power_split_with_lca"
DOE_RESULTS_FOLDER_PATH_SPLIT_BETTER_CELLS = (
    pathlib.Path(__file__).parent / "results_doe_power_split_better_cells_with_lca"
)
DOE_RESULTS_FOLDER_PATH_SPLIT_BETTER_CELLS_OK_LIFESPAN = (
    pathlib.Path(__file__).parent / "results_doe_power_split_better_cells_with_lca_ok_lifespan"
)
WORKDIR_FOLDER_PATH = pathlib.Path(__file__).parent / "workdir"


def test_sizing_sr22_with_lca():
    # TODO: Check why he needs the propulsion data as inputs
    # ANS: still used for the Z_cg of the aircraft which is assumed to have only a minor influence

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_with_lca.xml"
    process_file_name = "full_sizing_fuel_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        1601.0, rel=1e-2
    )
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(992.0, rel=1e-2)
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        234.00, rel=1e-2
    )


def test_sizing_sr22_electric_two_motors_improved():
    """
    Same tests as above except we test some possible improvements:
    - Not certify the aircraft under IFR but rather VFR, in our context it means a reserve of 30
        min instead of 45 min
    - Keep decreasing the range to 50 nm (only 23% of flights) (not worth not done) but instead
        we'll look at the evolution of MTOW and battery mass with range
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_electric_two_motors_with_lca.xml"
    process_file_name = "full_sizing_electric_two_motors_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val("data:mission:sizing:main_route:reserve:duration", units="min", val=30.0)
    problem.set_val("data:TLAR:range", units="NM", val=100.0)

    problem.run_model()
    problem.write_outputs()


def test_doe_sr22_hybrid_power_split():
    """
    Tests a hybrid sr22 with the same climb, cruise, descent and reserve profile as the original
    one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights) for various power
    split.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid_with_lca.xml"
    process_file_name = "full_sizing_hybrid_with_lca.yml"

    power_splits = np.linspace(80, 95, 16)

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Model options are set up straight into the configuration file
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}
    problem.setup()

    for power_split in power_splits:
        problem.set_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
            power_split,
            units="percent",
        )
        problem.run_model()

        problem.output_file_path = DOE_RESULTS_FOLDER_PATH_SPLIT / (
            str(int(power_split)) + "_percent_split_mda_with_lca.xml"
        )
        problem.write_outputs()


def test_doe_sr22_hybrid_power_split_better_cells():
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
    xml_file_name = "input_sr22_hybrid_with_lca.xml"
    process_file_name = "full_sizing_hybrid_with_lca.yml"

    power_splits = np.arange(70.0, 79.5, 1)
    # power_splits = [67]

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

    # For now not changing the battery lifespan, it's actually closer to 600 cycles than 700 cycles.

    for power_split in power_splits:
        problem.set_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
            power_split,
            units="percent",
        )
        problem.run_model()

        problem.output_file_path = DOE_RESULTS_FOLDER_PATH_SPLIT_BETTER_CELLS / (
            str(int(power_split)) + "_percent_split_mda.xml"
        )
        problem.write_outputs()


def test_doe_sr22_hybrid_power_split_better_cells_ok_lifespan():
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
    xml_file_name = "input_sr22_hybrid_with_lca.xml"
    process_file_name = "full_sizing_hybrid_with_lca.yml"

    power_splits = np.arange(70.0, 95.5, 1)
    # power_splits = [67]

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
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan",
        val=600.0,
        units=None,
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

    # For now not changing the battery lifespan, it's actually closer to 600 cycles than 700 cycles.

    for power_split in power_splits:
        problem.set_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
            power_split,
            units="percent",
        )
        problem.run_model()

        problem.output_file_path = DOE_RESULTS_FOLDER_PATH_SPLIT_BETTER_CELLS_OK_LIFESPAN / (
            str(int(power_split)) + "_percent_split_mda.xml"
        )
        problem.write_outputs()


def test_optimization_sr22_hybrid_with_lca():
    """
    Optimizes the hybrid sr22 with the same climb, cruise, descent and reserve profile as the o
    riginal one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights).
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid_with_lca.xml"
    process_file_name = "full_sizing_hybrid_with_lca.yml"

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
        lower=70.0,
        upper=95.0,
    )
    # We can't go lower than 69.0% otherwise we start requiring electric motor for which the rated
    # power is above what our model can handle.
    problem.model.add_objective(name="data:environmental_impact:single_score", scaler=1e7)

    recorder = om.SqliteRecorder("driver_cases_better_cells_with_lca.sql")
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
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan",
        val=600.0,
        units=None,
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
    problem.output_file_path = RESULTS_FOLDER_PATH / "optim_power_split_with_lca.xml"
    problem.write_outputs()
