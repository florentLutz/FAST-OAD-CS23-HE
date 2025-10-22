import os
from shutil import rmtree, copy
import logging
from pathlib import Path
import pytest
from fastoad import api

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = Path(__file__).parent / "results"


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_sizing_atr_42_turboshaft():
    """Test the overall aircraft design process for a thermal-powered ATR 42."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    configurator = api.FASTOADProblemConfigurator(DATA_FOLDER_PATH / "atr42_turboshaft.yml")
    problem = configurator.get_problem()

    problem.write_needed_inputs(DATA_FOLDER_PATH / "atr42_inputs.xml")
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

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    problem.write_outputs()


def test_sizing_atr_42_retrofit_hybrid():
    """
    Test the overall aircraft design process with retrofitting the parallel hybrid powertrain in
    the ATR 42. In this configuration, the electrical component of the powertrain supplies power
    only during the climb phase of the flight mission.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    configurator = api.FASTOADProblemConfigurator(
        DATA_FOLDER_PATH / "oad_process_parallel_retrofit.yml"
    )
    problem = configurator.get_problem()

    problem.write_needed_inputs(DATA_FOLDER_PATH / "hybrid_atr_inputs_retrofit.xml")
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 30e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.setup()

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    problem.write_outputs()


def test_hybrid_atr_42_full_sizing():
    """
    Test the overall aircraft design process for parallel hybrid configuration. In this
    configuration, the electrical component of the powertrain supplies power only during the climb
    phase of the flight mission.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    configurator = api.FASTOADProblemConfigurator(
        DATA_FOLDER_PATH / "oad_process_parallel_full_sizing.yml"
    )
    problem = configurator.get_problem()

    # Load inputs
    copy(
        DATA_FOLDER_PATH / "resized_hybrid_inputs.xml",
        RESULTS_FOLDER_PATH / "resized_hybrid_inputs.xml",
    )

    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 30e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.setup()

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    problem.write_outputs()
