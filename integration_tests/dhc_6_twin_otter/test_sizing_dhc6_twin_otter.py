# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import pathlib
import logging

import pytest

import openmdao.api as om
import fastoad.api as oad

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
WORKDIR_FOLDER_PATH = pathlib.Path(__file__).parent / "workdir"


def test_sizing_dhc6_twin_otter():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_dhc6_twin_otter.xml"
    process_file_name = "full_sizing_dhc6_twin_otter.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    n2_path = RESULTS_FOLDER_PATH / "n2_dhc6_twin_otter.html"
    # api.list_modules(DATA_FOLDER_PATH /  process_file_name, force_text_output=True)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(name="data:geometry:wing:area", units="m**2", val=40.0)

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(5652, rel=1e-2)
    # Actual value is 5670 kg (-0.3%)
    assert problem.get_val("data:weight:aircraft:MLW", units="kg") == pytest.approx(
        5530.7, rel=1e-2
    )
    # Actual value is 5579 kg (-0.9%)
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        3326.3, rel=1e-2
    )
    # Actual value is 3320 kg (+0.2%)
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(783.1, rel=1e-2)
    # Actual value is 808 kg (-3.1%)


def test_sizing_twin_otter_pemfc_h2():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_h2_pemfc_twin_otter.xml"
    process_file_name = "full_sizing_pemfc_h2_twin_otter.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    n2_path = RESULTS_FOLDER_PATH / "n2_dhc6_twin_otter_h2.html"
    # api.list_modules(DATA_FOLDER_PATH /  process_file_name, force_text_output=True)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(name="data:geometry:wing:area", units="m**2", val=40.0)

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(5588, rel=1e-2)
    assert problem.get_val("data:weight:aircraft:MLW", units="kg") == pytest.approx(
        5553.2, rel=1e-2
    )
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        4384.7, rel=1e-2
    )
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(114.7, rel=1e-2)
