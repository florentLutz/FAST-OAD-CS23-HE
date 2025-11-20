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

from fastga_he.gui.payload_range import payload_range_outer
from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
DOE_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_doe"
DOE_RESULTS_FOLDER_PATH_SPLIT = pathlib.Path(__file__).parent / "results_doe_power_split"
WORKDIR_FOLDER_PATH = pathlib.Path(__file__).parent / "workdir"


@pytest.fixture(scope="module")
def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree(WORKDIR_FOLDER_PATH, ignore_errors=True)


def test_sizing_sr22():
    # TODO: Check why he needs the propulsion data as inputs
    # ANS: still used for the Z_cg of the aircraft which is assumed to have only a minor influence

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22.xml"
    process_file_name = "full_sizing_fuel.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_cirrus.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # om.n2(problem, show_browser=False, outfile=n2_path)

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


def test_sizing_sr22_op_mission():
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "full_sizing_out.xml"
    process_file_name = "op_mission_fuel.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / xml_file_name
    rename_variables_for_payload_range(ref_inputs)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val("data:mission:operational:range", val=200, units="NM")
    # problem.set_val("data:mission:operational:cruise:v_tas", val=144, units="knot")

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val("data:mission:operational:fuel", units="kg") == pytest.approx(
        68.35, rel=1e-2
    )


def test_sizing_sr22_electric_original():
    """
    Tests an electric sr22 with the same climb, cruise, descent and reserve profile as the original
    one but a range of 100 nm. The only change is the ratio between stall speed and reserve speed
    which is set at 1.3 (ratio between approach and stall)
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_electric.xml"
    process_file_name = "full_sizing_electric.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_sizing_sr22_electric_two_motors():
    """
    Same tests as above except, to increase confidence in e-motor results we will take two "stacked"
    motors instead of a big one, as is doable with EMRAX motor. They will be coupled by a gearbox
    of 0 kg and 0.99 efficiency. Also, because the E-motor doesn't lose power with altitude, the
    climb rate at cruise level will be improved which should result in better energy efficiency.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_electric_two_motors.xml"
    process_file_name = "full_sizing_electric_two_motors.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        2873.0, rel=1e-2
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(664.0, abs=1e-2)


def test_sizing_sr22_electric_two_motors_improved_range_sensitivity():
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
    xml_file_name = "input_sr22_electric_two_motors.xml"
    process_file_name = "full_sizing_electric_two_motors.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val("data:mission:sizing:main_route:reserve:duration", units="min", val=30.0)

    # Baseline
    # design_ranges = np.array([100])
    design_ranges = np.array([75, 100, 125, 150, 175, 180])

    mtows = []
    batt_masses = []

    # om.n2(problem, show_browser=False, outfile=n2_path)

    for design_range in design_ranges:
        problem.set_val("data:TLAR:range", units="NM", val=design_range)

        problem.run_model()

        _, _, residuals = problem.model.get_nonlinear_vectors()
        residuals = filter_residuals(residuals)

        problem.output_file_path = (
            RESULTS_FOLDER_PATH / "full_sizing_elec_out_two_motors_improved.xml"
        )
        problem.write_outputs()

        mtows.append(problem.get_val("data:weight:aircraft:MTOW", units="kg")[0])
        batt_masses.append(
            2.0
            * problem.get_val(
                "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
            )[0]
        )

    print("MTOW", mtows)
    print("Battery pack mass", batt_masses)


def test_sizing_sr22_electric_two_motors_improved_plus_range_sensitivity():
    """
    Same tests as above except we replace the cell with a high energy density cell. In that case we
    will take a cell from the company Amprius with a gravimetric energy density of 395 Wh/kg,
    whereas before it was 261.4 Wh/kg
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_electric_two_motors.xml"
    process_file_name = "full_sizing_electric_two_motors.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 1.34,
        "cell_weight_ref": 11.7e-3,
        "reference_curve_current": [100.0, 1000.0, 3000.0, 5100.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }

    problem.setup()

    problem.set_val("data:mission:sizing:main_route:reserve:duration", units="min", val=30.0)

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=4.0,
        units="h**-1",
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
        val=4.0,
        units="h**-1",
    )

    # Baseline
    design_ranges = np.array([200])
    # design_ranges = np.array([50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375])

    mtows = []
    batt_masses = []

    # om.n2(problem, show_browser=False, outfile=n2_path)

    for design_range in design_ranges:
        problem.set_val("data:TLAR:range", units="NM", val=design_range)
        problem.run_model()

        problem.output_file_path = (
            RESULTS_FOLDER_PATH / "full_sizing_elec_out_two_motors_plus_improved.xml"
        )
        problem.write_outputs()

        mtows.append(problem.get_val("data:weight:aircraft:MTOW", units="kg")[0])
        batt_masses.append(
            2.0
            * problem.get_val(
                "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
            )[0]
        )

        print("MTOW", mtows)
        print("Battery pack mass", batt_masses)


def test_op_mission_with_target_improved_plus():
    # Now we do the same for the "plus" version, a priori we could reuse the same "problem" just
    # change the inputs
    # ref_electric_process_file_name = "op_mission_electric.yml"
    ref_electric_process_file_name = "payload_range_electric.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / ref_electric_process_file_name)
    problem_electric_high_bed = configurator.get_problem()

    ref_inputs_high_bed_electric = (
        RESULTS_FOLDER_PATH / "full_sizing_elec_out_two_motors_plus_improved.xml"
    )
    rename_variables_for_payload_range(ref_inputs_high_bed_electric)

    # Add a threshold SoC
    datafile = oad.DataFile(ref_inputs_high_bed_electric)
    datafile.append(
        oad.Variable(name="data:mission:operational:threshold_SoC", val=0.0, units="percent")
    )
    datafile.save()

    problem_electric_high_bed.write_needed_inputs(ref_inputs_high_bed_electric)
    problem_electric_high_bed.read_inputs()

    problem_electric_high_bed.model_options["*"] = {
        "cell_capacity_ref": 1.34,
        "cell_weight_ref": 11.7e-3,
        "reference_curve_current": [100.0, 1000.0, 3000.0, 5100.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }

    problem_electric_high_bed.setup()

    problem_electric_high_bed.run_model()
    problem_electric_high_bed.write_outputs()


def test_payload_range_comparison():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Run the payload range estimation for the thermal case
    thermal_xml_file_name = "full_sizing_out.xml"
    thermal_process_file_name = "payload_range_thermal.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / thermal_process_file_name)
    problem_thermal = configurator.get_problem()

    # Create inputs
    ref_inputs_thermal = RESULTS_FOLDER_PATH / thermal_xml_file_name
    rename_variables_for_payload_range(ref_inputs_thermal)

    problem_thermal.write_needed_inputs(ref_inputs_thermal)
    problem_thermal.read_inputs()
    problem_thermal.setup()
    problem_thermal.run_model()
    problem_thermal.write_outputs()

    # Now do the same for the first electric (with reference cell)
    ref_electric_xml_file_name = "full_sizing_elec_out_two_motors_improved.xml"
    ref_electric_process_file_name = "payload_range_electric.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / ref_electric_process_file_name)
    problem_electric_ref = configurator.get_problem()

    # Create inputs
    ref_inputs_ref_electric = RESULTS_FOLDER_PATH / ref_electric_xml_file_name
    rename_variables_for_payload_range(ref_inputs_ref_electric)

    # Add a threshold SoC
    datafile = oad.DataFile(ref_inputs_ref_electric)
    datafile.append(
        oad.Variable(name="data:mission:payload_range:threshold_SoC", val=0.0, units="percent")
    )
    datafile.save()

    problem_electric_ref.write_needed_inputs(ref_inputs_ref_electric)
    problem_electric_ref.read_inputs()
    problem_electric_ref.setup()
    problem_electric_ref.run_model()
    problem_electric_ref.write_outputs()

    # Now we do the same for the "plus" version, a priori we could reuse the same "problem" just
    # change the inputs
    problem_electric_high_bed = configurator.get_problem()

    ref_inputs_high_bed_electric = (
        RESULTS_FOLDER_PATH / "full_sizing_elec_out_two_motors_plus_improved.xml"
    )
    rename_variables_for_payload_range(ref_inputs_high_bed_electric)

    # Add a threshold SoC
    datafile = oad.DataFile(ref_inputs_high_bed_electric)
    datafile.append(
        oad.Variable(name="data:mission:payload_range:threshold_SoC", val=0.0, units="percent")
    )
    datafile.save()

    problem_electric_high_bed.write_needed_inputs(ref_inputs_high_bed_electric)
    problem_electric_high_bed.read_inputs()

    problem_electric_high_bed.model_options["*"] = {
        "cell_capacity_ref": 1.34,
        "cell_weight_ref": 11.7e-3,
        "reference_curve_current": [100.0, 1000.0, 3000.0, 5100.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }

    problem_electric_high_bed.setup()

    problem_electric_high_bed.run_model()
    problem_electric_high_bed.output_file_path = (
        RESULTS_FOLDER_PATH / "payload_range_elec_out_plus.xml"
    )
    problem_electric_high_bed.write_outputs()

    fig = payload_range_outer(
        aircraft_file_path=problem_thermal.output_file_path, name="Cirrus SR22"
    )
    fig = payload_range_outer(
        aircraft_file_path=problem_electric_ref.output_file_path,
        name="Electric SR22 with reference cell",
        fig=fig,
    )
    fig = payload_range_outer(
        aircraft_file_path=problem_electric_high_bed.output_file_path,
        name="Electric SR22 with high BED cell",
        fig=fig,
    )
    fig.show()


def test_sizing_sr22_hybrid():
    """
    Tests a hybrid sr22 with the same climb, cruise, descent and reserve profile as the original
    one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights).
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
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}

    problem.setup()

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_out_mda.xml"
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
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid.yml"

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
        lower=80.0,
        upper=95.0,
    )
    problem.model.add_objective(name="data:environmental_impact:sizing:emissions", units="kg")

    recorder = om.SqliteRecorder("driver_cases.sql")
    problem.driver.add_recorder(recorder)

    problem.driver.recording_options["record_desvars"] = True
    problem.driver.recording_options["record_objectives"] = True

    problem.setup()
    problem.run_driver()
    problem.write_outputs()


def test_sizing_sr22_hybrid_new_bed():
    """
    Tests a hybrid sr22 with the same climb, cruise, descent and reserve profile as the original
    one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights).
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    # process_file_name = "full_sizing_hybrid.yml"
    process_file_name = "full_sizing_hybrid_fixed_eff.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}
    problem.model_options["*"] = {"cell_weight_ref": 50.0e-3 * 261.0 / 350.0}

    problem.setup()

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.set_val(
        name="data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        units="percent",
        val=36.0,
    )

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_out_mda_new_bed.xml"
    problem.write_outputs()


def test_optimization_sr22_hybrid_new_bed():
    """
    Optimizes the hybrid sr22 with the same climb, cruise, descent, and reserve profile as the
    original one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights).
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
        lower=40.0,
        upper=80.0,
    )
    problem.model.add_objective(name="data:environmental_impact:sizing:emissions", units="kg")

    recorder = om.SqliteRecorder("driver_cases.sql")
    problem.driver.add_recorder(recorder)

    problem.driver.recording_options["record_desvars"] = True
    problem.driver.recording_options["record_objectives"] = True

    # To assess the range of BED for which hybrid is feasible, we artificially improve the battery
    # energy density by reducing the weight of the reference cell. Reference cell has a battery
    # energy density of 261 Wh/kg with a cell weight of 50.0g we do simple cross product to achieve
    # energy densities of 275, 300, 325, ...
    problem.model_options["*"] = {"cell_weight_ref": 50.0e-3 * 261.0 / 400.0}

    problem.setup()

    problem.set_val(
        name="data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        units="percent",
        val=70.0,
    )

    problem.run_driver()
    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_out_new_bed.xml"
    problem.write_outputs()


def test_optimization_sr22_hybrid_new_bed_fixed_eff():
    """
    This test optimizes the hybrid sr22 with the same climb, cruise, descent, and reserve profile as the
    original one, but with a range of 200 nm (this represents 75% of all Cirrus SR22 flights). The battery
    energy density is changed to see for which value a full electric design becomes preferable. Assuming
    the motor efficiency stays the same, the model shows that with today’s technology and setup,
    efficiency will degrade FAST !
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid_fixed_eff.yml"

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
        lower=25.0,
        upper=40.0,
    )
    problem.model.add_objective(name="data:environmental_impact:sizing:emissions", units="kg")

    recorder = om.SqliteRecorder("driver_cases.sql")
    problem.driver.add_recorder(recorder)

    problem.driver.recording_options["record_desvars"] = True
    problem.driver.recording_options["record_objectives"] = True

    # To assess the range of BED for which hybrid is feasible, we artificially improve the battery
    # energy density by reducing the weight of the reference cell. Reference cell has a battery
    # energy density of 261 Wh/kg with a cell weight of 50.0g we do simple cross product to achieve
    # energy densities of 275, 300, 325, ...
    problem.model_options["*"] = {"cell_weight_ref": 50.0e-3 * 261.0 / 400.0}

    problem.setup()

    problem.set_val(
        name="data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        units="percent",
        val=30.0,
    )

    problem.run_driver()
    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_out_new_bed_fixed_eff.xml"
    problem.write_outputs()


def test_optimization_sr22_hybrid_new_bed_fixed_eff_rotax():
    """
    This test optimizes the hybrid sr22 with the same climb, cruise, descent, and reserve profile as the
    original one, but with a range of 200 nm (this represents 75% of all Cirrus SR22 flights). The battery
    energy density is changed to see for which value a full electric design becomes preferable. Assuming
    the motor efficiency stays the same, the model shows that with today’s technology and setup,
    efficiency will degrade FAST ! Plus a Rotax will be used because at some point
    the original Lycoming is not suited anymore.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid_fixed_eff_rotax.yml"

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
        lower=10.0,
        upper=30.0,
    )
    problem.model.add_objective(name="data:environmental_impact:sizing:emissions", units="kg")

    recorder = om.SqliteRecorder("driver_cases.sql")
    problem.driver.add_recorder(recorder)

    problem.driver.recording_options["record_desvars"] = True
    problem.driver.recording_options["record_objectives"] = True

    # To assess the range of BED for which hybrid is feasible, we artificially improve the battery
    # energy density by reducing the weight of the reference cell. Reference cell has a battery
    # energy density of 261 Wh/kg with a cell weight of 50.0g we do simple cross product to achieve
    # energy densities of 275, 300, 325, ...
    problem.model_options["*"] = {"cell_weight_ref": 50.0e-3 * 261.0 / 475.0}

    problem.setup()

    problem.set_val(
        name="data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        units="percent",
        val=24.0,
    )

    problem.run_driver()
    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_out_new_bed_fixed_eff.xml"
    problem.write_outputs()


def test_sizing_sr22_hybrid_new_bed_fixed_eff_rotax():
    """
    This test sizes the hybrid sr22 with the same climb, cruise, descent, and reserve profile as the
    original one, but with a range of 200 nm (this represents 75% of all Cirrus SR22 flights). The battery
    energy density is changed to see for which value a full electric design becomes preferable. Assuming
    the motor efficiency stays the same, the model shows that with today’s technology and setup,
    efficiency will degrade FAST ! Plus a Rotax will be used because at some point
    the original Lycoming is not suited anymore.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid_fixed_eff_rotax.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # To assess the range of BED for which hybrid is feasible, we artificially improve the battery
    # energy density by reducing the weight of the reference cell. Reference cell has a battery
    # energy density of 261 Wh/kg with a cell weight of 50.0g we do simple cross product to achieve
    # energy densities of 275, 300, 325, ...
    problem.model_options["*"] = {"cell_weight_ref": 50.0e-3 * 261.0 / 475.0}

    problem.setup()

    problem.set_val(
        name="data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        units="percent",
        val=18.0,
    )

    problem.run_model()
    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "full_sizing_hybrid_out_fixed_eff_mda_rotax_mda.xml"
    )
    problem.write_outputs()


def test_sizing_electric_sr22_fixed_eff_sensitivity_bed():
    """
    To judge when the full electric becomes more worth that the hybrid, we'll look at when the
    full electric becomes feasible. Fixed efficiency for motor and inverter will be assumed,
    and we'll gradually increase the bed of the reference cell. Full electric will be assumed
    feasible if the weight of the aircraft isn't more than double the original design or when
    the derivative of the MTOW wrt the design range isn't too high. What is too high ? I don't know
    yet. According to the formula 19 proposed by Hepperle et al. it is 5.3 kg/nm. According to this
    criteria it only occurs for BED of 600 Wh/kg or higher. We'll instead consider this to be
    feasible if MTOW < MTOWeSR22 defined earlier which happens for BED of 475 Wh/kg
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_electric_fixed_eff.xml"
    process_file_name = "full_sizing_electric_fixed_eff.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # To assess the range of BED for which hybrid is feasible, we artificially improve the battery
    # energy density by reducing the weight of the reference cell. Reference cell has a battery
    # energy density of 261 Wh/kg with a cell weight of 50.0g we do simple cross product to achieve
    # energy densities of 275, 300, 325, ...
    problem.model_options["*"] = {"cell_weight_ref": 50.0e-3 * 261.0 / 475.0}

    problem.setup()

    problem.run_model()
    problem.write_outputs()

    # I'm afraid If I just put +1 nm the code will converge without changing the results
    problem.set_val("data:TLAR:range", units="NM", val=210)
    problem.run_model()

    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_elec_out_fixed_eff_plus_5.xml"
    problem.write_outputs()


def test_optimization_sr22_hybrid_slsqp():
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
    problem.model.approx_totals(step=1e-7)

    problem.model.add_design_var(
        name="data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        units="percent",
        lower=80.0,
        upper=95.0,
    )
    # We expect emissions around 260kg of CO2eq
    problem.model.add_objective(
        name="data:environmental_impact:sizing:emissions", units="g", scaler=1e-4
    )

    problem.driver = om.ScipyOptimizeDriver(
        tol=1e-4, optimizer="SLSQP", debug_print=["objs", "desvars"]
    )
    problem.driver.opt_settings["eps"] = 1e-7

    recorder = om.SqliteRecorder("driver_cases_slsqp.sql")
    problem.driver.add_recorder(recorder)

    problem.driver.recording_options["record_desvars"] = True
    problem.driver.recording_options["record_objectives"] = True

    problem.setup()

    problem.set_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        val=90.0,
        units="percent",
    )

    problem.run_driver()
    problem.write_outputs()


def test_sizing_sr22_hybrid_power_share():
    """
    Tests a hybrid sr22 with the same climb, cruise, descent and reserve profile as the original
    one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights).
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid_power_share.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Model options are set up straight into the configuration file
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }
    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}

    problem.setup()

    # For smooth init
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=20.0
    )

    # For the problem to run but without touching the source file as it is shared
    problem.set_val(
        "data:propulsion:he_power_train:PMSM:motor_1:rpm_rating", val=4000.0, units="min**-1"
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_out_power_share_mda.xml"
    problem.write_outputs()


def test_doe_sr22_hybrid_power_share():
    """
    Runs the hybrid sr22 with the same climb, cruise, descent and reserve profile as the
    original one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights) for various
    power shares.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid_power_share.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Model options are set up straight into the configuration file
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }
    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}

    problem.setup()

    # For smooth init
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=20.0
    )

    # For the problem to run but without touching the source file as it is shared
    problem.set_val(
        "data:propulsion:he_power_train:PMSM:motor_1:rpm_rating", val=4000.0, units="min**-1"
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )

    # power_shares = np.linspace(150, 190, 18)
    power_shares = np.linspace(154.7059, 157.0588, 5)[1:-1]

    for power_share in power_shares:
        problem.set_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_share",
            power_share,
            units="kW",
        )
        problem.run_model()

        problem.output_file_path = DOE_RESULTS_FOLDER_PATH / (
            str(int(power_share)) + "_kW_power_share_mda.xml"
        )
        problem.write_outputs()


def test_optimization_sr22_hybrid_power_share():
    """
    Optimizes the hybrid sr22 with the same climb, cruise, descent and reserve profile as the
    original one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights).
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid_power_share.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.model.approx_totals()

    problem.model.add_design_var(
        name="data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_share",
        units="kW",
        lower=160.0,
        upper=195.0,
    )
    problem.model.add_objective(name="data:environmental_impact:sizing:emissions", units="kg")

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }
    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}

    problem.setup()

    # For smooth init
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=20.0
    )

    # For the problem to run but without touching the source file as it is shared
    problem.set_val(
        "data:propulsion:he_power_train:PMSM:motor_1:rpm_rating", val=4000.0, units="min**-1"
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )
    problem.run_driver()

    problem.write_outputs()


def test_sizing_sr22_hybrid_no_lto():
    """
    Tests a hybrid sr22 with the same climb, cruise, descent and reserve profile as the original
    one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights) and a battery sized
    for no LTO emissions.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_hybrid.xml"
    process_file_name = "full_sizing_hybrid.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()
    problem.input_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_in_no_LTO.xml"

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)

    # Here we wet the power split so that it is 100% aimed towards the electric circuit when we are
    # in the LTO cycle (<3000ft) and towards the electric motor the rest of the time. The number of
    # points below 3000ft was counted by hand on the results of the previous hybrid run. Two points
    # were added for taxi. It will be checked a posteriori. Here is the bearkdown:
    # - 1 point for taxi out
    # - 10 points for start of climb
    # - 52 points for rest of climb, cruise and start of descent
    # - 8 points for rest of descent
    # - 20 points for reserve
    # - 1 point for taxi in
    power_split = 100.0 * np.concatenate(
        (np.zeros(12), np.ones(61), np.zeros(8), np.ones(10), np.zeros(1))
    )
    datafile = oad.DataFile(problem.input_file_path)
    datafile[
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split"
    ].value = power_split
    datafile.save()

    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
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
        val=8.0,
        units="h**-1",
    )

    problem.run_model()

    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_hybrid_out_no_LTO.xml"
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)
    # Also rename the .csv so they are not overwritten (because the conf file and pt watcher files
    # are shared).
    pathlib.Path(RESULTS_FOLDER_PATH / "hybrid_propulsion_no_LTO.csv").unlink()
    mission_file = pathlib.Path(RESULTS_FOLDER_PATH / "hybrid_propulsion.csv")
    mission_file.rename(RESULTS_FOLDER_PATH / "hybrid_propulsion_no_LTO.csv")

    pathlib.Path(RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher_no_LTO.csv").unlink()
    pt_watcher_file = pathlib.Path(RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher.csv")
    pt_watcher_file.rename(RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher_no_LTO.csv")


def test_sizing_sr22_hybrid_no_lto_improved():
    """
    Previous tests reveals that the battery is sized for power and still has 55% SOC at the end of
    the mission so we effectively have a dead, unused weight. This tests smoothes the transition
    between electric and thermal power to use that energy.
    """
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
    smoothing_zone_width = 17
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

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
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
        val=8.0,
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
    mission_file = pathlib.Path(RESULTS_FOLDER_PATH / "hybrid_propulsion.csv")
    mission_file.rename(RESULTS_FOLDER_PATH / "hybrid_propulsion_no_LTO_improved.csv")

    if pathlib.Path(
        RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher_no_LTO_improved.csv"
    ).exists():
        pathlib.Path(
            RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher_no_LTO_improved.csv"
        ).unlink()
    pt_watcher_file = pathlib.Path(RESULTS_FOLDER_PATH / "hybrid_propulsion_pt_watcher.csv")
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
