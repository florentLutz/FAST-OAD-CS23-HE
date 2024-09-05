# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
from shutil import rmtree
import logging

import pytest

import openmdao.api as om
import fastoad.api as oad

from fastga.utils.postprocessing.analysis_and_plots import (
    mass_breakdown_bar_plot,
    aircraft_geometry_plot,
)

from utils.filter_residuals import filter_residuals

from fastga_he.gui.power_train_network_viewer import power_train_network_viewer
from fastga_he.gui.power_train_weight_breakdown import power_train_mass_breakdown

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
    ) == pytest.approx(1275.0, abs=1.0)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        5.81, abs=1e-2
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


def test_ecopulse_new_wing_pt_mass_breakdown():
    path_to_result_file = pth.join(RESULTS_FOLDER_PATH, "oad_process_outputs_ecopulse_new_wing.xml")
    path_to_pt_file = pth.join(DATA_FOLDER_PATH, "ecopulse_powertrain_new_wing.yml")

    fig = power_train_mass_breakdown(path_to_result_file, path_to_pt_file)
    fig.update_layout(uniformtext=dict(minsize=28, mode="hide"))
    fig.update_traces(textfont=dict(family=["Arial Black", "Arial"], size=[30]))
    fig.show()


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

    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(321.0, abs=1.0)


def test_weight_comparison():
    ref_aircraft = pth.join(RESULTS_FOLDER_PATH, "oad_process_outputs_op.xml")
    ref_aircraft_sizing = pth.join(RESULTS_FOLDER_PATH, "oad_process_outputs_ref.xml")
    ref_datafile = oad.DataFile(ref_aircraft_sizing)

    # Add fake values corresponding to the op mission, don't want to do it directly in the file
    # as it might get overwritten
    datafile = oad.DataFile(ref_aircraft)

    tow_value = datafile["data:mission:operational:TOW"].value[0]
    tow_units = datafile["data:mission:operational:TOW"].units
    datafile.append(oad.Variable("data:weight:aircraft:MTOW", val=tow_value, units=tow_units))

    payload_value = datafile["data:mission:operational:payload:mass"].value[0]
    payload_units = datafile["data:mission:operational:payload:mass"].units
    datafile.append(
        oad.Variable("data:weight:aircraft:payload", val=payload_value, units=payload_units)
    )

    fuel_value = datafile["data:mission:operational:fuel"].value[0]
    fuel_units = datafile["data:mission:operational:fuel"].units
    datafile.append(oad.Variable("data:mission:sizing:fuel", val=fuel_value, units=fuel_units))

    datafile.append(ref_datafile["data:weight:airframe:mass"])
    datafile.append(ref_datafile["data:weight:furniture:mass"])
    datafile.append(ref_datafile["data:weight:propulsion:mass"])
    datafile.append(ref_datafile["data:weight:systems:mass"])

    datafile.save()

    retrofit_aircraft = pth.join(RESULTS_FOLDER_PATH, "oad_process_outputs_ecopulse_retrofit.xml")

    retrofit_datafile = oad.DataFile(retrofit_aircraft)
    retrofit_datafile["data:weight:airframe:mass"].value = ref_datafile[
        "data:weight:airframe:mass"
    ].value
    retrofit_datafile.save()

    new_wing_aircraft = pth.join(RESULTS_FOLDER_PATH, "oad_process_outputs_ecopulse_new_wing.xml")

    fig = mass_breakdown_bar_plot(
        ref_aircraft, name="Reference aircraft on the operational mission"
    )
    fig = mass_breakdown_bar_plot(retrofit_aircraft, name="Conservative hybrid design", fig=fig)
    fig = mass_breakdown_bar_plot(new_wing_aircraft, name="Hybrid design from scratch", fig=fig)
    fig.update_layout(font=dict(size=20))
    fig.show()


def test_geometry_comparison():
    ref_aircraft_sizing = pth.join(RESULTS_FOLDER_PATH, "oad_process_outputs_ref.xml")
    new_wing_aircraft = pth.join(RESULTS_FOLDER_PATH, "oad_process_outputs_ecopulse_new_wing.xml")

    fig = aircraft_geometry_plot(ref_aircraft_sizing, name="Reference aircraft")
    fig = aircraft_geometry_plot(new_wing_aircraft, name="Hybrid design from scratch", fig=fig)
    fig.show()
