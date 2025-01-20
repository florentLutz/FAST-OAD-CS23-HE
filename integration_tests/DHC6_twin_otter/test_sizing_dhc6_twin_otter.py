# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
import os
import os.path as pth
from shutil import rmtree
import logging
import numpy as np
import csv
import pytest

import openmdao.api as om
import fastoad.api as oad

from utils.filter_residuals import filter_residuals

from fastga_he.gui.power_train_network_viewer import power_train_network_viewer
from fastga_he.gui.power_train_weight_breakdown import power_train_mass_breakdown

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


def residuals_analyzer(recorder_path):
    cr = om.CaseReader(recorder_path)

    solver_cases = cr.get_cases("root.nonlinear_solver")

    # Get only the last 10 cases (or all if less than 10)
    last_10_cases = solver_cases[-10:]

    variable_dict = {name: 0.0 for name in last_10_cases[-1].residuals}

    for case in last_10_cases:
        for residual in case.residuals:
            variable_dict[residual] = np.sum(np.abs(case.residuals[residual]))

    sorted_variable_dict = dict(sorted(variable_dict.items(), key=lambda x: x[1], reverse=True))

    return sorted_variable_dict


def outputs_analyzer(recorder_path):
    cr = om.CaseReader(recorder_path)

    solver_cases = cr.get_cases("root.nonlinear_solver")

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
    rmtree("D:/tmp", ignore_errors=True)


def test_sizing_dhc6_twin_otter():

    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_dhc6_twin_otter.xml"
    process_file_name = "full_sizing_dhc6_twin_otter.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_dhc6_twin_otter.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()
    """

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        5670.0, rel=5e-2
    )
    assert problem.get_val("data:weight:aircraft:MLW", units="kg") == pytest.approx(
        5579.0, rel=5e-2
    )
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        3121.0, rel=5e-2
    )
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        1163.00, rel=5e-2
    )
    """


def test_operational_mission_dhc6_twin_otter():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_dhc6_twin_otter_op_mission.xml"
    process_file_name = "operational_mission_dhc6_twin_otter.yml"

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
        5670.0, rel=5e-2
    )
    assert problem.get_val("data:mission:operational:fuel", units="kg") == pytest.approx(
        1163.00, rel=5e-2
    )

    assert problem.get_val(
        "data:environmental_impact:operational:fuel_emissions", units="kg"
    ) == pytest.approx(4271.75, rel=1e-2)


def test_pemfc_h2_gas_tank_powertrain_network():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "pemfc_h2_propulsion.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "pemfc_h2_propulsion.html")

    if not pth.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def test_pemfc_h2_gas_retrofit():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_pemfc_h2_dhc6.xml"
    process_file_name = "pemfc_h2_gas_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=40.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_pemfc_wing_pod_h2_gas_tank_powertrain_network():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "pemfc_wing_pod_h2_propulsion.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "pemfc_wing_pod_h2_propulsion.html")

    if not pth.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def test_pemfc_wing_pod_h2_gas_retrofit():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_pemfc_wing_pod_h2_dhc6.xml"
    process_file_name = "pemfc_wing_pod_h2_gas_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=40.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_pemfc_belly_h2_gas_tank_wingpod_powertrain_network():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "pemfc_belly_h2_tank_wingpod_propulsion.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "pemfc_belly_h2_tank_wingpod_propulsion.html")

    if not pth.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def test_pemfc_belly_h2_gas_tank_wingpod_retrofit():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_pemfc_belly_h2_tank_wingpod_dhc6.xml"
    process_file_name = "pemfc_belly_h2_gas_tank_wingpod_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    # Copy the value from source file
    problem.set_val(name="data:geometry:wing:area", units="m**2", val=40.72394)

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_turboshaft_pemfc_hybrid_powertrain_network():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "turboshaft_pemfc_hybrid_propulsion.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "turboshaft_pemfc_hybrid_propulsion.html")

    if not pth.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def test_turboshaft_pemfc_hybrid_retrofit_residual_check():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_turboshaft_pemfc_hybrid_dhc6.xml"
    process_file_name = "pemfc_turboprop_hybrid_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    recorder_path = pth.join(RESULTS_FOLDER_PATH, "dhc6_hybrid_cases.sql")
    recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True

    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=40.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()
    sorted_variable_residuals = residuals_analyzer(recorder_path)
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


def test_turboshaft_pemfc_hybrid_retrofit_output_check():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_turboshaft_pemfc_hybrid_dhc6.xml"
    process_file_name = "pemfc_turboprop_hybrid_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    recorder_path = pth.join(RESULTS_FOLDER_PATH, "dhc6_hybrid_cases.sql")
    recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_outputs"] = True

    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=40.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()
    sorted_variable_residuals = outputs_analyzer(recorder_path)
    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    # Construct the file path
    file_path = os.path.join(RESULTS_FOLDER_PATH, "dhc6_hybrid_outputs_analysis.csv")

    # Open the file for writing
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(["Variable name", "Outputs"])

        # Write the sum of residuals for each iteration
        for name, out in sorted_variable_residuals.items():
            writer.writerow([name, out])


def test_turboshaft_pemfc_hybrid_retrofit():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_turboshaft_pemfc_hybrid_dhc6.xml"
    process_file_name = "pemfc_turboprop_hybrid_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=40.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)
    print(residuals)

    problem.write_outputs()


def test_pemfc_lh2_tank_powertrain_network():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "pemfc_lh2_propulsion.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "pemfc_lh2_propulsion.html")

    if not pth.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def test_pemfc_lh2_retrofit():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_pemfc_lh2_dhc6.xml"
    process_file_name = "pemfc_lh2_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}
    problem.model_options["*tank_weight_lh2*"] = {"structure_factor": 1.43}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=40.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_pemfc_wing_pod_lh2_tank_powertrain_network():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "pemfc_wing_pod_lh2_propulsion.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "pemfc_wing_pod_lh2_propulsion.html")

    if not pth.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def test_pemfc_wing_pod_lh2_retrofit():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_pemfc_wing_pod_lh2_dhc6.xml"
    process_file_name = "pemfc_wing_pod_lh2_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}
    problem.model_options["*nusselt_number*"] = {"position": "underbelly"}
    problem.model_options["*heat_radiation*"] = {"position": "underbelly"}
    problem.model_options["*tank_weight_lh2*"] = {"structure_factor": 1.43}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=40.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_pemfc_belly_lh2_tank_wingpod_powertrain_network():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "pemfc_belly_lh2_tank_wingpod_propulsion.yml")
    network_file_path = pth.join(
        RESULTS_FOLDER_PATH, "pemfc_belly_lh2_tank_wingpod_propulsion.html"
    )

    if not pth.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def test_pemfc_belly_lh2_tank_wingpod_retrofit():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_pemfc_belly_lh2_tank_wingpod_dhc6.xml"
    process_file_name = "pemfc_belly_lh2_tank_wingpod_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}
    problem.model_options["*nusselt_number*"] = {"position": "wing_pod"}
    problem.model_options["*heat_radiation*"] = {"position": "wing_pod"}
    problem.model_options["*tank_weight_lh2*"] = {"structure_factor": 1.43}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=40.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_pemfc_lh2_hybrid_retrofit():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_turboshaft_pemfc_lh2_hybrid_dhc6.xml"
    process_file_name = "pemfc_turboprop_hybrid_lh2_resize.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}
    problem.model_options["*tank_weight_lh2*"] = {"structure_factor": 1.43}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=5000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=40.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)
    print(residuals)

    problem.write_outputs()


def test_ghc_6():

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_ghc6.xml"
    process_file_name = "full_sizing_ghc6.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=6000.0)
    problem.set_val(
        name="data:geometry:wing:area", units="m**2", val=50.72394
    )  # Copy the value from source file

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()
    MLW = problem.get_val("data:weight:aircraft:MLW", units="kg")
    MTOW = problem.get_val("data:weight:aircraft:MTOW", units="kg")

    print("\n=========== MTOW ===========")
    print(MTOW)
    print("\n=========== MLW ===========")
    print(MLW)
