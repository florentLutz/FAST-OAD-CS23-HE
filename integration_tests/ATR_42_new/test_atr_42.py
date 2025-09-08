import os
import shutil
from shutil import rmtree
import logging
import os.path as pth
import numpy as np
import openmdao.api as om
import pandas as pd
import pytest
from fastoad.module_management._plugins import FastoadLoader
from numpy.testing import assert_allclose
import fastoad.api as oad
from fastoad import api
from fastoad.io import VariableIO
from fastoad.io.configuration.configuration import FASTOADProblemConfigurator
import csv

from fastga_he.gui.power_train_network_viewer import power_train_network_viewer


from utils.filter_residuals import filter_residuals

FastoadLoader()


DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
CONFIGURATION_FILE = "oad_process_parallel_retrofit.yml"
MISSION_FILE = "sizing_mission_R.yml"
SOURCE_FILE = "inputs_parallel_retrofit.xml"
RESULTS_FOLDER = "problem_folder"


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_sizing_atr_42_turboshaft():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "atr42_inputs_turboshaft.xml"
    process_file_name = "atr42_turboshaft.yml"

    configurator = api.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name))
    # print(oad.RegisterSubmodel.active_models["service.mass.propulsion"])

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_ATR42.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()
    model = problem.model

    # Give good initial guess on a few key value to reduce the time it takes to converge

    problem.set_val("data:geometry:wing:MAC:at25percent:x", units="m", val=10.0)
    problem.set_val("data:geometry:wing:area", units="m**2", val=54.5)
    problem.set_val("data:geometry:horizontal_tail:area", units="m**2", val=10.645)
    problem.set_val("data:geometry:vertical_tail:area", units="m**2", val=11.0)

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=18600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=11250.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=16700.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=18300.0)

    problem.set_val("data:weight:aircraft_empty:mass", units="kg", val=11414.2)
    problem.set_val("data:weight:aircraft_empty:CG:x", units="m", val=10.514757)

    problem.set_val(
        "subgroup.performances.solve_equilibrium.update_mass.mass",
        units="kg",
        val=np.linspace(18000, 16000, 90),
    )

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    # Construct the file path
    file_path = os.path.join(RESULTS_FOLDER_PATH, "cases.csv")

    problem.write_outputs()


def test_sizing_atr_42_retrofit_hybrid():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    configurator = api.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, "oad_process_parallel_retrofit.yml"))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, "inputs_parallel_hybrid_retrofit.xml")
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_ATR42.html")

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 30e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.setup()
    model = problem.model

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )
    # Give good initial guess on a few key value to reduce the time it takes to converge

    problem.set_val("data:geometry:wing:MAC:at25percent:x", units="m", val=10.0)
    problem.set_val("data:geometry:wing:area", units="m**2", val=53.3)
    problem.set_val("data:geometry:horizontal_tail:area", units="m**2", val=10.645)
    problem.set_val("data:geometry:vertical_tail:area", units="m**2", val=11.0)

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=18600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=11250.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=16700.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=18300.0)

    problem.set_val("data:weight:aircraft_empty:mass", units="kg", val=11414.2)
    problem.set_val("data:weight:aircraft_empty:CG:x", units="m", val=10.514757)

    problem.set_val(
        "performances.solve_equilibrium.update_mass.mass",
        units="kg",
        val=np.linspace(18000, 16000, 90),
    )

    datafile = oad.DataFile("data/atr42_retrofit_data.xml")

    list_of_variables_to_set = [
        "data:weight:airframe:wing:mass",
        "data:weight:airframe:fuselage:mass",
        "data:weight:airframe:horizontal_tail:mass",
        "data:weight:airframe:vertical_tail:mass",
        "data:weight:airframe:landing_gear:main:mass",
        "data:weight:airframe:landing_gear:front:mass",
        "data:propulsion:he_power_train:mass",
        "data:weight:systems:auxiliary_power_unit:mass",
        "data:weight:systems:electric_systems:electric_generation:mass",
        "data:weight:systems:electric_systems:electric_common_installation:mass",
        "data:weight:systems:hydraulic_systems:mass",
        "data:weight:systems:fire_protection:mass",
        "data:weight:systems:flight_furnishing:mass",
        "data:weight:systems:automatic_flight_system:mass",
        "data:weight:systems:communications:mass",
        "data:weight:systems:ECS:mass",
        "data:weight:systems:de-icing:mass",
        "data:weight:systems:navigation:mass",
        "data:weight:systems:flight_controls:mass",
        "data:weight:furniture:furnishing:mass",
        "data:weight:furniture:water:mass",
        "data:weight:furniture:interior_integration:mass",
        "data:weight:furniture:insulation:mass",
        "data:weight:furniture:cabin_lighting:mass",
        "data:weight:furniture:seats_crew_accommodation:mass",
        "data:weight:furniture:oxygen:mass",
        "data:weight:operational:items:passenger_seats:mass",
        "data:weight:operational:items:unusable_fuel:mass",
        "data:weight:operational:items:documents_toolkit:mass",
        "data:weight:operational:items:galley_structure:mass",
        "data:weight:operational:equipment:others:mass",
    ]

    for names in list_of_variables_to_set:
        problem.set_val(names, datafile[names].value, units=datafile[names].units)

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    problem.write_outputs()

def test_hybrid_atr_42_full_sizing():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True


    configurator = api.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, "oad_process_parallel_fullsizing.yml"))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, "inputs_parallel_hybrid_fullsizing.xml")
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_ATR42.html")

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()
    model = problem.model

    problem.set_val("data:geometry:wing:MAC:at25percent:x", units="m", val=10.0)
    problem.set_val("data:geometry:wing:area", units="m**2", val=53.3)
    problem.set_val("data:geometry:horizontal_tail:area", units="m**2", val=10.645)
    problem.set_val("data:geometry:vertical_tail:area", units="m**2", val=11.0)
    #

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=18000.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=11659.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=17159.5)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=18189.10294470838)

    problem.set_val("data:weight:aircraft_empty:mass", units="kg", val=11414.2)
    problem.set_val("data:weight:aircraft_empty:CG:x", units="m", val=10.514757)

    problem.set_val(
        "subgroup.performances.solve_equilibrium.update_mass.mass",
        units="kg",
        val=np.linspace(18000, 16000, 90),
    )

    datafile = oad.DataFile("data/atr42_retrofit_data.xml")

    list_of_variables_to_set = [
        "data:weight:airframe:wing:mass",
        "data:weight:airframe:fuselage:mass",
        "data:weight:airframe:horizontal_tail:mass",
        "data:weight:airframe:vertical_tail:mass",
        "data:weight:airframe:landing_gear:main:mass",
        "data:weight:airframe:landing_gear:front:mass",
        "data:propulsion:he_power_train:mass",
        "data:weight:systems:auxiliary_power_unit:mass",
        "data:weight:systems:electric_systems:electric_generation:mass",
        "data:weight:systems:electric_systems:electric_common_installation:mass",
        "data:weight:systems:hydraulic_systems:mass",
        "data:weight:systems:fire_protection:mass",
        "data:weight:systems:flight_furnishing:mass",
        "data:weight:systems:automatic_flight_system:mass",
        "data:weight:systems:communications:mass",
        "data:weight:systems:ECS:mass",
        "data:weight:systems:de-icing:mass",
        "data:weight:systems:navigation:mass",
        "data:weight:systems:flight_controls:mass",
        "data:weight:furniture:furnishing:mass",
        "data:weight:furniture:water:mass",
        "data:weight:furniture:interior_integration:mass",
        "data:weight:furniture:insulation:mass",
        "data:weight:furniture:cabin_lighting:mass",
        "data:weight:furniture:seats_crew_accommodation:mass",
        "data:weight:furniture:oxygen:mass",
        "data:weight:operational:items:passenger_seats:mass",
        "data:weight:operational:items:unusable_fuel:mass",
        "data:weight:operational:items:documents_toolkit:mass",
        "data:weight:operational:items:galley_structure:mass",
        "data:weight:operational:equipment:others:mass",
    ]

    for names in list_of_variables_to_set:
        problem.set_val(names, datafile[names].value, units=datafile[names].units)


    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    problem.write_outputs()