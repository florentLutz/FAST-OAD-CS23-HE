import os
from shutil import rmtree
import logging
import os.path as pth
import numpy as np
import openmdao.api as om
import pytest
import fastoad.api as oad
from fastoad import api

from utils.filter_residuals import filter_residuals
from constant import ELECTRIC_ATR_42_VARIABLE_INITIALIZATION

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_sizing_atr_42_turboshaft():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "atr42_inputs.xml"
    process_file_name = "atr42_turboshaft.yml"

    configurator = api.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

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

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    problem.write_outputs()


def test_sizing_atr_42_retrofit_hybrid():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    configurator = api.FASTOADProblemConfigurator(
        pth.join(DATA_FOLDER_PATH, "oad_process_parallel_retrofit.yml")
    )
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, "hybrid_atr_inputs_retrofit.xml")
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

    problem.set_val(
        "performances.solve_equilibrium.update_mass.mass",
        units="kg",
        val=np.linspace(18000, 16000, 90),
    )

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.write_outputs()


def test_hybrid_atr_42_full_sizing():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    configurator = api.FASTOADProblemConfigurator(
        pth.join(DATA_FOLDER_PATH, "oad_process_parallel_full_sizing.yml")
    )
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, "hybrid_atr_inputs_full_sizing.xml")

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 30e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.setup()

    problem.set_val(
        "subgroup.performances.solve_equilibrium.update_mass.mass",
        units="kg",
        val=np.linspace(18000, 16000, 90),
    )

    datafile = oad.DataFile("data/set_up_data_full_sizing.xml")

    for names in ELECTRIC_ATR_42_VARIABLE_INITIALIZATION:
        problem.set_val(names, datafile[names].value, units=datafile[names].units)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    problem.write_outputs()
