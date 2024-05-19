# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
from shutil import rmtree
import logging

import pytest

import openmdao.api as om
import fastoad.api as oad

from utils.filter_residuals import filter_residuals

from fastga_he.gui.power_train_network_viewer import power_train_network_viewer

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


@pytest.fixture(scope="module")
def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def test_sizing_tbm900():

    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_tbm900.xml"
    process_file_name = "full_sizing_tbm900.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_tbm900.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.model_options["*propeller_1*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3378.0, rel=1e-2
    )
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        2089.0, rel=1e-2
    )
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        802.00, rel=1e-2
    )


def test_operational_mission_tbm_900():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_tbm900_op_mission.xml"
    process_file_name = "operational_mission_tbm900.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:mission:operational:TOW", units="kg") == pytest.approx(
        3012.0, rel=1e-2
    )
    assert problem.get_val("data:mission:operational:fuel", units="kg") == pytest.approx(
        273.00, abs=1
    )
    assert problem.get_val(
        "data:environmental_impact:operational:fuel_emissions", units="kg"
    ) == pytest.approx(1042.0, rel=1e-2)


def test_ecopulse_powertrain_network():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "turbo_electric_propulsion.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "turbo_electric_propulsion.html")

    if not pth.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def test_retrofit_ecopulse():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_ecopulse.xml"
    process_file_name = "ecopulse_retrofit.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_1*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(334.0, abs=1.0)
    assert problem.get_val("data:propulsion:he_power_train:mass", units="kg") == pytest.approx(
        829.0, abs=1.0
    )
    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(1277.0, abs=1.0)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        5.82, abs=1e-2
    )


def test_ecopulse_new_wing():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_ecopulse_new_wing.xml"
    process_file_name = "ecopulse_new_wing.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_1*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(319.0, abs=1.0)
    assert problem.get_val("data:propulsion:he_power_train:mass", units="kg") == pytest.approx(
        726.0, abs=1.0
    )
    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(1218.0, abs=1.0)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        4.492, abs=1e-2
    )
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        2473.5, rel=1e-3
    )


def test_ecopulse_new_wing_mission_analysis():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "ecopulse_new_wing_mission_analysis.xml"
    process_file_name = "ecopulse_new_wing_mission_analysis.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_1*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(327.0, abs=1.0)
