from pathlib import Path
import logging
from fastoad import api
from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = Path(__file__).parent / "results"


def test_sizing_atr_72():
    """Test the overall aircraft design process for a thermal-powered ATR 72."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    configurator = api.FASTOADProblemConfigurator(
        DATA_FOLDER_PATH / "atr_72_full_sizing.yml"
    )
    problem = configurator.get_problem()

    problem.write_needed_inputs(DATA_FOLDER_PATH / "inputs_full_sizing.xml")
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=22000.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=10000.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=20000.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=21850.0)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()
