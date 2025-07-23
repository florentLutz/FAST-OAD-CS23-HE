# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pathlib
from shutil import rmtree
import logging

import pytest

import fastoad.api as oad

from fastga_he.gui.payload_range import payload_range_outer
from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
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
    motors instead of a big one, as is doable with EMRAX engine. They will be coupled by a gearbox
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
        3172.0, rel=1e-2
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(745.0, abs=1e-2)


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
    problem.set_val("data:TLAR:range", units="NM", val=100.0)
    # problem.set_val("data:TLAR:range", units="NM", val=50.0)
    # problem.model.nonlinear_solver.options["rtol"] = 1e-8

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_elec_out_two_motors_improved.xml"
    problem.write_outputs()

    # Range tested
    # 50, 75, 100, 125, 150, 175, 180

    # Corresponding MTOW [kg]
    # 1891.0, 2148, 2453, 2886, 3507, 4679, 5043

    # Corresponding battery mass [kg]
    # 2.0 * 314, 2.0 * 398, 2.0 * 513, 2.0 * 670, 2.0 * 892, 2.0 * 1300, 2.0 * 1425


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
    problem.set_val("data:TLAR:range", units="NM", val=100.0)
    # problem.set_val("data:TLAR:range", units="NM", val=375.0)

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "full_sizing_elec_out_two_motors_plus_improved.xml"
    )
    problem.write_outputs()

    # Range tested
    # 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375

    # Corresponding MTOW [kg]
    # 1373, 1490, 1620, 1749, 1897, 2061, 2250, 2472, 2700, 2964, 3279, 3671, 4214, 5025

    # Corresponding battery mass [kg]
    # 2.0 * 127, 2.0 * 171, 2.0 * 218, 2.0 * 266, 2.0 * 320, 2.0 * 379, 2.0 * 445, 2.0 * 521,
    # 2.0 * 604, 2.0 * 699, 2.0 * 812, 2.0 * 951, 2.0 * 1141, 2.0 * 1421


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
    one but a range of 200 nm (this represents 75% of all Cirrus SR22 flights). The only change is
    the ratio between stall speed and reserve speed which is set at 1.3 (ratio between approach
    and stall)
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
    problem.setup()

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


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
    }

    datafile = oad.DataFile(source_file_path)

    for op_name, sizing_name in op_name_to_sizing_name.items():
        variable_to_add = oad.Variable(
            op_name, val=datafile[sizing_name].value, units=datafile[sizing_name].units
        )
        datafile.append(variable_to_add)

    datafile.save()
