# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import pathlib
import logging
from shutil import copy
import pytest

import openmdao.api as om
import fastoad.api as oad

from fastga_he.gui.power_train_network_viewer import power_train_network_viewer
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

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(5674, rel=1e-2)
    # Actual value is 5670 kg (+0.07%)
    assert problem.get_val("data:weight:aircraft:MLW", units="kg") == pytest.approx(
        5537.7, rel=1e-2
    )
    # Actual value is 5579 kg (-0.8%)
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(3333, rel=1e-2)
    # Actual value is 3320 kg (+0.4%)
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(798.7, rel=1e-2)
    # Actual value is 808 kg (-1.2%)


def test_retrofit_twin_otter_pemfc_h2():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_h2_pemfc_twin_otter_retrofit.xml"
    process_file_name = "retrofit_pemfc_h2_twin_otter.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    n2_path = RESULTS_FOLDER_PATH / "n2_dhc6_twin_otter_h2_retrofit.html"

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)
    print(residuals)
    problem.write_outputs()

def test_retrofit_twin_otter_pemfc_h2_hybrid():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_h2_pemfc_twin_otter_retrofit_hybrid.xml"
    process_file_name = "retrofit_pemfc_h2_twin_otter_hybrid.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    n2_path = RESULTS_FOLDER_PATH / "n2_dhc6_twin_otter_h2_hybrid_retrofit.html"

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)
    print(residuals)
    problem.write_outputs()

def test_hybrid_dhc_6_hybrid_powertrain_network():
    pt_file_path = DATA_FOLDER_PATH / "turboshaft_pemfc_hybrid_propulsion_retrofit.yml"
    network_file_path = RESULTS_FOLDER_PATH / "dhc_6_h2_hybrid.html"

    power_train_network_viewer(
        pt_file_path,
        network_file_path,
        animated_plot=True,
        plot_scaling=1.3,
        legend_position="BR",
        legend_scaling=1.3,
        from_propulsor=False
    )


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

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=6000.0)
    problem.set_val(name="data:geometry:wing:area", units="m**2", val=50.0)

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)
    print(residuals)
    problem.write_outputs()


def test_hybrid_dhc_6_powertrain_network():
    pt_file_path = DATA_FOLDER_PATH / "turboshaft_pemfc_hybrid_propulsion.yml"
    network_file_path = RESULTS_FOLDER_PATH / "dhc_6_h2.html"

    power_train_network_viewer(
        pt_file_path,
        network_file_path,
        animated_plot=False,
        orientation="LR",
        plot_scaling=1.3,
        legend_position="BR",
        legend_scaling=1.3,
    )


def test_sizing_twin_otter_pemfc_h2_inside():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_h2_pemfc_twin_otter_inside.xml"
    process_file_name = "full_sizing_pemfc_h2_twin_otter_inside.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    n2_path = RESULTS_FOLDER_PATH / "n2_dhc6_twin_otter_h2_inside.html"
    # api.list_modules(DATA_FOLDER_PATH /  process_file_name, force_text_output=True)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=6000.0)
    problem.set_val(name="data:geometry:wing:area", units="m**2", val=50.0)

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_sizing_twin_otter_pemfc_h2_with_bop():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_h2_pemfc_twin_otter_with_bop.xml"
    process_file_name = "full_sizing_pemfc_h2_twin_otter_with_bop.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    n2_path = RESULTS_FOLDER_PATH / "n2_dhc6_twin_otter_h2_with_bop.html"
    # api.list_modules(DATA_FOLDER_PATH /  process_file_name, force_text_output=True)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    # Load inputs
    copy(
        DATA_FOLDER_PATH / "input_h2_pemfc_twin_otter_with_bop.xml",
        RESULTS_FOLDER_PATH / "oad_process_inputs_pemfc_h2_gas_with_bop.xml",
    )

    # problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    print(residuals.keys())

    problem.write_outputs()


def test_sizing_twin_otter_pemfc_h2_simplified():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_h2_pemfc_twin_otter_bop_simplified.xml"
    process_file_name = "full_sizing_pemfc_h2_twin_otter_bop_simplified.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    n2_path = RESULTS_FOLDER_PATH / "n2_dhc6_twin_otter_h2_inside.html"
    # api.list_modules(DATA_FOLDER_PATH /  process_file_name, force_text_output=True)

    problem.model_options["*propeller_*"] = {"mass_as_input": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val(name="data:weight:aircraft:MTOW", units="kg", val=6000.0)
    problem.set_val(name="data:geometry:wing:area", units="m**2", val=50.0)

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()
