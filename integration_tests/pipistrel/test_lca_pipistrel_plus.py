#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib

import logging

import pytest

import numpy as np

import fastoad.api as oad

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_lca_pipistrel_reference_cell():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%, with the reference cell.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca.xml"
    process_file_name = "pipistrel_configuration_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # To ensure consistency with previous Pipistrel results
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:min_safe_SOC",
        units="percent",
        val=7.5,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:min_safe_SOC",
        units="percent",
        val=7.5,
    )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()


def test_lca_pipistrel_plus_reference_cell():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%, with the reference cell.
    - The front battery pack will be used only for reserve, while the back one will be used for the
        nominal mission. The LCA will be run with the battery aging model (so the two battery should
        age very differently) but the effect of SOH on performances won't be considered.
    - The LCA configuration file will be automatically generated.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_plus_source_with_lca.xml"
    process_file_name = "pipistrel_plus_configuration_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*battery_pack_1*"] = {
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

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:aging:cyclic_effect_k_factor",
        val=0.0,
    )

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # To ensure consistency with previous Pipistrel results
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:min_safe_SOC",
        units="percent",
        val=7.5,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:min_safe_SOC",
        units="percent",
        val=7.5,
    )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()


def test_lca_pipistrel_plus_plus_high_bed_cell():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%, with the reference cell.
    - The front battery pack will be used only for reserve, while the back one will be used for the
        nominal mission. The LCA will be run with the battery aging model (so the two battery should
        age very differently) but the effect of SOH on performances won't be considered.
    - The front battery will use a high energy density cell
    - The LCA configuration file won't be automatically generated.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_plus_source_with_lca.xml"
    process_file_name = "pipistrel_plus_plus_configuration_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*battery_pack_1*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.model_options["*battery_pack_2*"] = {
        "cell_capacity_ref": 1.34,
        "cell_weight_ref": 11.7e-3,
        "reference_curve_current": [100.0, 1000.0, 3000.0, 5100.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }

    problem.setup()

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
        val=4.0,
        units="h**-1",
    )

    # TODO: This case will work because it is reasonable to assume that the high BED cell has the
    # TODO: same polarization curve as the reference cell otherwise we would have needed to change
    # TODO: the submodels for the polarization curve but since they are shared it would have caused
    # TODO: a problem.

    # Also we assume same aging mechanism
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:aging:cyclic_effect_k_factor",
        val=0.0,
    )

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # To ensure consistency with previous Pipistrel results
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:min_safe_SOC",
        units="percent",
        val=7.5,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:min_safe_SOC",
        units="percent",
        val=7.5,
    )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()


def test_only_lca_pipistrel_reference_cell_pessimistic():
    """
    Tests that contains:
    - An LCA evaluation of the reference Pipistrel with a buy to fly ratio of 2 for composite and
        a European electric mix
    - OAD results from the first test will be used.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_out_with_lca.xml"
    process_file_name = "pipistrel_configuration_only_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val("data:environmental_impact:buy_to_fly:composite", val=2.0)

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.0036902, rel=1e-3
    )


def test_only_lca_pipistrel_plus_plus_pessimistic():
    """
    Tests that contains:
    - An LCA evaluation of the Pipistrel plus plus with a buy to fly ratio of 2 for composite and
        a European electric mix
    - OAD results from the test_lca_pipistrel_plus_plus_high_bed_cell will be used.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_plus_plus_out_with_lca.xml"
    process_file_name = "pipistrel_plus_plus_configuration_only_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val("data:environmental_impact:buy_to_fly:composite", val=2.0)

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00352396, rel=1e-3
    )
