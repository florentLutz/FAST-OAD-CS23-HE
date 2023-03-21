import os
import os.path as pth
import logging

import pytest

import openmdao.api as om
import fastoad.api as oad

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


def test_dummy():

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)

    # Define used files depending on options
    xml_file_name = "full_sizing.xml"
    process_file_name = "full_sizing.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1000.0)
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_fuel_and_battery():

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)

    # Define used files depending on options
    xml_file_name = "full_sizing_fuel_and_battery.xml"
    process_file_name = "full_sizing_fuel_and_battery.yml"

    # configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    #
    # # Create inputs
    # ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    # configurator.write_needed_inputs(ref_inputs)
    #
    # # Create problems with inputs
    # problem = configurator.get_problem(read_inputs=True)
    # problem.setup()
    # problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1200.0)
    # problem.run_model()

    oad.generate_inputs(
        pth.join(DATA_FOLDER_PATH, process_file_name),
        pth.join(DATA_FOLDER_PATH, xml_file_name),
        overwrite=True,
    )
    problem = oad.evaluate_problem(pth.join(DATA_FOLDER_PATH, process_file_name), overwrite=True)

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_mission_vector_from_yml_fuel_and_battery():

    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"
    process_file_name = "mission_vector_fuel_and_battery_propulsion.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(0.0, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(140.08, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(0.06715826, abs=1e-2)


def test_sizing_sr22():

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)

    # Define used files depending on options
    xml_file_name = "input_sr22.xml"
    process_file_name = "full_sizing_fuel.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem, show_browser=True)

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1000.0)
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()
