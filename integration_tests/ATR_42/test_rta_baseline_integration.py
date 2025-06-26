import os.path as pth
import shutil
from shutil import rmtree
import logging

import numpy as np
import openmdao.api as om
import pandas as pd
import pytest
from fastoad.module_management._plugins import FastoadLoader
from numpy.testing import assert_allclose

from fastoad import api
from fastoad.io import VariableIO
from fastoad.io.configuration.configuration import FASTOADProblemConfigurator

from utils.filter_residuals import filter_residuals

FastoadLoader()


DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
CONFIGURATION_FILE = "oad_process_float_performance_from_he.yml"
MISSION_FILE = "sizing_mission_R.yml"
SOURCE_FILE = "problem_outputs.xml"
RESULTS_FOLDER = "problem_folder"


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_non_regression_mission(cleanup):
    run_non_regression_test(
        CONFIGURATION_FILE,
        SOURCE_FILE,
        RESULTS_FOLDER,
        check_only_mtow=False,
        tolerance=1.0e-2,
    )


def run_non_regression_test(
    conf_file,
    legacy_result_file,
    result_dir,
    check_only_mtow=True,
    tolerance=5.0e-2,
):
    results_folder_path = pth.join(RESULTS_FOLDER_PATH, result_dir)
    configuration_file_path = pth.join(results_folder_path, conf_file)

    # Copy of configuration file and generation of problem instance ------------------
    api.generate_configuration_file(
        configuration_file_path, distribution_name="rta"
    )  # just ensure folders are created...
    shutil.copy(pth.join(DATA_FOLDER_PATH, conf_file), configuration_file_path)
    shutil.copy(
        pth.join(DATA_FOLDER_PATH, MISSION_FILE),
        pth.join(results_folder_path, MISSION_FILE),
    )
    configurator = FASTOADProblemConfigurator(configuration_file_path)

    # Generation and reading of inputs ----------------------------------------
    ref_inputs = pth.join(DATA_FOLDER_PATH, legacy_result_file)
    configurator.write_needed_inputs(ref_inputs)
    problem = configurator.get_problem()
    problem.read_inputs()
    problem.setup()

    # Run model ---------------------------------------------------------------
    problem.run_model()
    problem.write_outputs()

    try:
        problem.model.performance.flight_points.to_csv(
            pth.join(results_folder_path, "flight_points.csv"),
            sep="\t",
            decimal=",",
        )
    except AttributeError:
        pass
    om.view_connections(
        problem,
        outfile=pth.join(results_folder_path, "connections.html"),
        show_browser=False,
    )

    # Check that weight-performances loop correctly converged
    assert_allclose(
        problem["data:weight:aircraft:OWE"],
        problem["data:weight:airframe:mass"]
        + problem["data:weight:propulsion:mass"]
        + problem["data:weight:systems:mass"]
        + problem["data:weight:furniture:mass"]
        + problem["data:weight:operational:mass"],
        atol=1,
    )
    assert_allclose(
        problem["data:weight:aircraft:MZFW"],
        problem["data:weight:aircraft:OWE"] + problem["data:weight:aircraft:max_payload"],
        atol=1,
    )

    ref_var_list = VariableIO(
        pth.join(DATA_FOLDER_PATH, legacy_result_file),
    ).read()

    row_list = []
    for ref_var in ref_var_list:
        try:
            value = problem.get_val(ref_var.name, units=ref_var.units)[0]
        except KeyError:
            continue
        row_list.append(
            {
                "name": ref_var.name,
                "units": ref_var.units,
                "ref_value": ref_var.value[0],
                "value": value,
            }
        )

    df = pd.DataFrame(row_list)
    df["rel_delta"] = (df.value - df.ref_value) / df.ref_value
    df.loc[(df.ref_value == 0) & (abs(df.value) <= 1e-10), "rel_delta"] = 0.0
    df["abs_rel_delta"] = np.abs(df.rel_delta)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 120)
    print(df.sort_values(by=["abs_rel_delta"]))
"""
    if check_only_mtow:
        assert np.all(df.abs_rel_delta.loc[df.name == "data:weight:aircraft:MTOW"] < tolerance)
    else:
        assert np.all(df.abs_rel_delta < tolerance)
"""


def test_sizing_atr_42():

    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = SOURCE_FILE
    process_file_name = CONFIGURATION_FILE

    configurator = api.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_ATR72.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=23000.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=15000.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=20000.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=21000.0)

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    ###