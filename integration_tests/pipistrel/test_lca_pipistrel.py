#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib

import logging

import pytest

import fastoad.api as oad

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_lca"


def test_lca_pipistrel_reference():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%
    - A LCA evaluation of the sized configuration without considering the effect of the SOH on the
        functional unit. The effect of the performances on the battery aging will however be
        considered without any tuning.
    - The LCA conf file will be automatically generated.
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

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00084237, rel=1e-3
    )


def test_lca_pipistrel_full_aging_effect():
    """
    Tests that contains:
    - A LCA evaluation of the sized configuration which considers the fact that, as the battery
        ages, its capacity decreases leading to a reduced number of FU. A separate code has been
        used to estimate how many flight hour can be carried out with one battery, the number of
        battery will thus be equal to the max airframe hour divided by that value.
    - The LCA conf file will not be automatically generated. Rather it will use the previous one
        and the value for battery mass and energy per FU will be inputs as the results of the
        previously mentioned code.
    - Similarly save for the aforementioned inputs, the same inputs will be used so the output file
        of the precedent sizing process will be used
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca_proper_aging.xml"
    process_file_name = "pipistrel_configuration_with_lca_proper_aging.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Check that the right input has been set, that is  the number of hours flown before a
    # replacement is required as computed by a separate script
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:hours_per_battery", units="h"
    ) == pytest.approx(354.94, rel=1e-5)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:hours_per_battery", units="h"
    ) == pytest.approx(354.94, rel=1e-5)

    # We use a different model to compute the mass per FU for the battery in that test and in the
    # previous one, these equivalent inputs give the same results, they are computed as:
    # hours_per_battery = main_route:duration * battery:lifespan.
    # So the implicit assumption is that the battery capacity doesn't fade with cycle.
    # problem.set_val(
    #     "data:propulsion:he_power_train:battery_pack:battery_pack_1:hours_per_battery",
    #     units="h",
    #     val=428.34,
    # )
    # problem.set_val(
    #     "data:propulsion:he_power_train:battery_pack:battery_pack_2:hours_per_battery",
    #     units="h",
    #     val=428.34,
    # )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00096246, rel=1e-3
    )
