# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
from shutil import rmtree
import logging
import numpy as np
import pytest
import os
import csv

import openmdao.api as om
import fastoad.api as oad

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
WORKDIR_FOLDER_PATH = pth.join(pth.dirname(__file__), "workdir")


def residuals_analyzer(recorder_path, solver):
    cr = om.CaseReader(recorder_path)

    solver_cases = cr.get_cases(solver)

    # Get only the last 10 cases (or all if less than 10)
    last_10_cases = solver_cases[-3:]

    variable_dict = {name: 0.0 for name in last_10_cases[-1].residuals}

    for case in last_10_cases:
        for residual in case.residuals:
            variable_dict[residual] = np.sum(np.abs(case.residuals[residual]))

    sorted_variable_dict = dict(sorted(variable_dict.items(), key=lambda x: x[1], reverse=True))

    return sorted_variable_dict
def outputs_analyzer(recorder_path, solver):
    cr = om.CaseReader(recorder_path)

    solver_cases = cr.get_cases(solver)

    # Get only the last 10 cases (or all if less than 10)
    last_10_cases = solver_cases[-10:]

    # Initialize a dictionary to store outputs for each variable and iteration
    variable_dict = {}

    for case in last_10_cases:
        for output, value in case.outputs.items():
            if (
                    isinstance(value, np.ndarray) and value.ndim == 1
            ):  # Check if the value is a 1D numpy array
                if output not in variable_dict:
                    variable_dict[output] = []
                # Extract the scalar value if it's a single-element array
                scalar_value = value.item() if value.size == 1 else value
                variable_dict[output].append(scalar_value)

    # Remove variables with all zero values
    non_zero_variable_dict = {
        key: value
        for key, value in variable_dict.items()
        if not np.allclose(np.array(value), 0, atol=1e-10)
    }

    return non_zero_variable_dict

@pytest.fixture(scope="module")
def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree(WORKDIR_FOLDER_PATH, ignore_errors=True)


def test_sizing_kodiak_100():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_kodiak100.xml"
    process_file_name = "full_sizing_kodiak100.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_kodiak100.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3280.0, rel=1e-2
    )
    # Actual value is 3290 kg
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        1727.0, rel=1e-2
    )
    # Actual value is 1712 kg
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        933.00, rel=1e-2
    )
    # Actual value is 2110 lbs or 960 kg


def test_sizing_kodiak_100_full_electric():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_elec_kodiak100.xml"
    process_file_name = "full_sizing_kodiak100_elec.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_kodiak100.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 5.0,
        "cell_weight_ref": 45.0e-3,
    }

    problem.setup()

    model = problem.model
    recorder = om.SqliteRecorder(pth.join(DATA_FOLDER_PATH, "cases.sql"))
    solver = model.aircraft_sizing.performances.solve_equilibrium.compute_dep_equilibrium.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True
    solver.recording_options["record_outputs"] = True

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    sorted_variable_residuals = residuals_analyzer(recorder, solver)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    # Construct the file path
    file_path = os.path.join(RESULTS_FOLDER_PATH, "cases.csv")

    # Open the file for writing
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(["Variable name", "Residuals"])

        # Write the sum of residuals for each iteration
        for name, sum_res in sorted_variable_residuals.items():
            writer.writerow([name, sum_res])

    cr = om.CaseReader(recorder)
    #
    problem.write_outputs()
    #
    # assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
    #     4174.0, rel=1e-2
    # )
    # # Actual value is 3290 kg
    # assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
    #     1727.0, rel=1e-2
    # )
    # # Actual value is 1712 kg
    # assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
    #     933.00, rel=1e-2
    # )
    # # Actual value is 2110 lbs or 960 kg

def test_read_case_recorder():
    recorder_data_file_path = pth.join(DATA_FOLDER_PATH, "cases.sql")

    cr = om.CaseReader(recorder_data_file_path)
    # sorted_variable_residuals = residuals_analyzer(recorder_data_file_path, solver)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    # Construct the file path
    file_path = os.path.join(RESULTS_FOLDER_PATH, "dhc6_hybrid_residuals_analysis.csv")

    # Open the file for writing
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(["Variable name", "Residuals"])

        # Write the sum of residuals for each iteration
        for name, sum_res in sorted_variable_residuals.items():
            writer.writerow([name, sum_res])

    cr = om.CaseReader(recorder_data_file_path)

def test_operational_mission_kodiak_100():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_kodiak100_op_mission.xml"
    process_file_name = "operational_mission_kodiak100.yml"

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
        3113.0, abs=1
    )
    assert problem.get_val("data:mission:operational:fuel", units="kg") == pytest.approx(
        246.00, abs=1
    )
    assert problem.get_val(
        "data:environmental_impact:operational:fuel_emissions", units="kg"
    ) == pytest.approx(938.0, abs=1)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(2.223, abs=1e-2)


def test_retrofit_hybrid_kodiak():
    """

    We'll take a new turboshaft that correspond to the PW206B as it seems to have a fairly good
    sfc according to https://en.wikipedia.org/wiki/Pratt_%26_Whitney_Canada_PW200. We'll use that
    reference sfc as well as some educated on OPR and thermodynamic to get the right k_sfc before
    we can get our hand on more data (possibly from Jane's). We'll consider that sfc is given at
    Sea Level Static with power equal to limit power. This gives an k_sfc of 1.11

    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_kodiak.xml"
    process_file_name = "hybrid_kodiak_retrofit.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # om.n2(problem, outfile=pth.join(RESULTS_FOLDER_PATH, "hybrid_kodiak_n2.html"))

    # Change battery pack characteristics so that they match those of a high power,
    # lower capacity cell like the Samsung INR18650-25R, we also take the weight fraction of the
    # Pipistrel battery. Assumes same polarization curve
    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.setup()

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(210.00, abs=1.0)
    assert problem.get_val("data:propulsion:he_power_train:mass", units="kg") == pytest.approx(
        529.0, abs=1.0
    )
    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(805.00, abs=1.0)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        1.924, abs=1e-2
    )


def test_retrofit_hybrid_kodiak_european_mix():
    """
    Computation of the emissions factor with the Europe electricity index.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "hybrid_kodiak_full_sizing.xml"
    process_file_name = "hybrid_kodiak_emissions_europe_mix.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(809.405, abs=1.0)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        1.933, abs=1e-2
    )


def test_retrofit_hybrid_kodiak_eu_mix_ft():
    """

    Computation of the emissions factor with the french electricity emission index and biofuel
    obtained with FT pathway.

    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "hybrid_kodiak_full_sizing.xml"
    process_file_name = "hybrid_kodiak_emissions_europe_mix_ft.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(76.19, rel=1e-3)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        0.182, rel=1e-3
    )


def test_operational_mission_kodiak_100_ft():
    """
    Computation of the emissions factor with the french electricity emission index and biofuel
    obtained with FT pathway.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "op_mission_full.xml"
    process_file_name = "op_kodiak_emissions_ft.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:operational:emissions", units="kg"
    ) == pytest.approx(81.505, abs=1.0)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(0.1930, abs=1e-2)


def test_retrofit_hybrid_kodiak_eu_mix_hefa():
    """

    Computation of the emissions factor with the french electricity emission index and biofuel
    obtained with HEFA pathway.

    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "hybrid_kodiak_full_sizing.xml"
    process_file_name = "hybrid_kodiak_emissions_europe_mix_hefa.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(202.01, rel=1e-3)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        0.4826, rel=1e-3
    )


def test_operational_mission_kodiak_100_hefa():
    """
    Computation of the emissions factor with the french electricity emission index and biofuel
    obtained with HEFA pathway.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "op_mission_full.xml"
    process_file_name = "op_kodiak_emissions_hefa.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:operational:emissions", units="kg"
    ) == pytest.approx(228.638, abs=1.0)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(0.5415, abs=1e-2)
