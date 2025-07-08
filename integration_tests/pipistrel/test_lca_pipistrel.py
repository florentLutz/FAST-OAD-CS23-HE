#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib

import logging

import pytest

import numpy as np
import openmdao.api as om

import fastoad.api as oad

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_lca"
SENSITIVITY_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_sensitivity_2"


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


def test_lca_pipistrel_full_aging_full_pipeline_90_percent():
    """
    Tests that contains:
     - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
         reference case), but with a starting SOC of 90%
     - A LCA evaluation of the sized configuration without considering the effect of the SOH on the
         functional unit. The effect of the performances on the battery aging will however be
         considered without any tuning.
     - The LCA conf file will be automatically generated.
     - Then said LCA wil be re-run with the proper effect of aging on the performances. Inputs for
         that second LCA will be the output of the first part of the test, as is done in the first
         two tests of that script. The only difference will be the number of hours flown per battery
         which will be a results of another script
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
    problem.input_file_path = RESULTS_FOLDER_PATH / "pipistrel_in_with_lca_90_percent.xml"
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start",
        units="percent",
        val=90.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:SOC_mission_start",
        units="percent",
        val=90.0,
    )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_90_percent.xml"
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00094497, rel=1e-3
    )

    # Now we create the second problem with the proper aging effect

    configurator_proper_aging = oad.FASTOADProblemConfigurator(
        DATA_FOLDER_PATH / "pipistrel_configuration_with_lca_proper_aging.yml"
    )
    problem_proper_aging = configurator_proper_aging.get_problem()

    # Create inputs
    ref_inputs = problem.output_file_path

    # Setup the problem
    problem_proper_aging.input_file_path = (
        RESULTS_FOLDER_PATH / "pipistrel_in_with_lca_90_percent_proper_aging.xml"
    )
    problem_proper_aging.write_needed_inputs(ref_inputs)
    problem_proper_aging.read_inputs()
    problem_proper_aging.setup()

    # Results from a separate script
    problem_proper_aging.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:hours_per_battery",
        units="h",
        val=365.25,
    )
    problem_proper_aging.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:hours_per_battery",
        units="h",
        val=365.25,
    )

    # Run the problem
    problem_proper_aging.run_model()

    # Write the outputs
    problem_proper_aging.output_file_path = (
        RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_90_percent_proper_aging.xml"
    )
    problem_proper_aging.write_outputs()

    assert problem_proper_aging.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.0010143792054993603, rel=1e-3
    )


def test_lca_pipistrel_full_aging_full_pipeline_80_percent():
    """
    Tests that contains:
     - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
         reference case), but with a starting SOC of 80%
     - A LCA evaluation of the sized configuration without considering the effect of the SOH on the
         functional unit. The effect of the performances on the battery aging will however be
         considered without any tuning.
     - The LCA conf file will be automatically generated.
     - Then said LCA wil be re-run with the proper effect of aging on the performances. Inputs for
         that second LCA will be the output of the first part of the test, as is done in the first
         two tests of that script. The only difference will be the number of hours flown per battery
         which will be a results of another script
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
    problem.input_file_path = RESULTS_FOLDER_PATH / "pipistrel_in_with_lca_80_percent.xml"
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start",
        units="percent",
        val=80.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:SOC_mission_start",
        units="percent",
        val=80.0,
    )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_80_percent.xml"
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00100379, rel=1e-3
    )

    # Now we create the second problem with the proper aging effect

    configurator_proper_aging = oad.FASTOADProblemConfigurator(
        DATA_FOLDER_PATH / "pipistrel_configuration_with_lca_proper_aging.yml"
    )
    problem_proper_aging = configurator_proper_aging.get_problem()

    # Create inputs
    ref_inputs = problem.output_file_path

    # Setup the problem
    problem_proper_aging.input_file_path = (
        RESULTS_FOLDER_PATH / "pipistrel_in_with_lca_80_percent_proper_aging.xml"
    )
    problem_proper_aging.write_needed_inputs(ref_inputs)
    problem_proper_aging.read_inputs()
    problem_proper_aging.setup()

    # Results from a separate script
    problem_proper_aging.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:hours_per_battery",
        units="h",
        val=373.82,
    )
    problem_proper_aging.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:hours_per_battery",
        units="h",
        val=373.82,
    )

    # Run the problem
    problem_proper_aging.run_model()

    # Write the outputs
    problem_proper_aging.output_file_path = (
        RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_80_percent_proper_aging.xml"
    )
    problem_proper_aging.write_outputs()

    assert problem_proper_aging.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00116834, rel=1e-3
    )


def test_lca_pipistrel_full_aging_effect_including_econ_degradation():
    """
    Tests that contains:
    - A LCA evaluation of the sized configuration which considers the fact that, as the battery
        ages, its capacity decreases leading to a reduced number of FU. A separate code has been
        used to estimate how many flight hour can be carried out with one battery, the number of
        battery will thus be equal to the max airframe hour divided by that value. This test also
        includes the fact that as battery ages flight time decreases so on average the aircraft
        spends more time in climb and descent leading to increased energy per time flown so more
        impact of the use phase
    - The LCA conf file will not be automatically generated. Rather it will use the previous one
        and the value for battery mass and energy per FU will be inputs as the results of the
        previously mentioned code.
    - Similarly save for the aforementioned inputs, the same inputs will be used so the output file
        of the precedent sizing process will be used
    - This will be done for several value of the lifespan so a plot can be shown.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca_proper_aging_with_econ.xml"
    process_file_name = "pipistrel_configuration_with_lca_proper_aging_with_econ.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    datafile = oad.DataFile(ref_inputs)

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Check that the right input has been set, that is  the number of hours flown before a
    # replacement is required as computed by a separate script
    hours_per_battery = 354.94
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:hours_per_battery", units="h"
    ) == pytest.approx(hours_per_battery, rel=1e-5)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:hours_per_battery", units="h"
    ) == pytest.approx(hours_per_battery, rel=1e-5)

    # Default value should give the same (inaccurate results) as the first step with aging effect
    # only on battery lifespan. However, we are interested here in the fact that since aging reduce
    # effective cruise range more energy is used per FU after the battery ages. Which depends on the
    # time since last battery renewal.

    # For reasons that should probably be investigated I can't access that value like the others,
    # but that should do it.
    airframe_hours = om.convert_units(
        datafile["data:TLAR:max_airframe_hours"].value[0],
        datafile["data:TLAR:max_airframe_hours"].units,
        "h",
    )
    # This will give us how long into the "life" of the battery
    time_since_last_renewal = airframe_hours % hours_per_battery
    print("Time since last renewal:", time_since_last_renewal, "h")

    # This regression which is based on the results of another script will give how many battery
    # cycle it corresponds to (not linear because of aging !) R2 of 0.999
    number_of_cycle_since_last_renewal = (
        8.62929561e-04 * time_since_last_renewal**2.0
        + 2.23157428 * time_since_last_renewal
        - 3.09920890
    )
    print(
        "Number of cycle since last renewal:",
        time_since_last_renewal,
        number_of_cycle_since_last_renewal,
    )

    # Computes the energy required to perform one hour of flight (which so happen to be one FU)
    # based on the results of another script
    energy_per_flight_hours = (
        2.32639768e-03 * number_of_cycle_since_last_renewal
        - 5.90621456 * number_of_cycle_since_last_renewal
        + 2.75735887e4
    )
    print("Energy for one hour flight:", energy_per_flight_hours, "W*h")

    problem.set_val(
        "data:LCA:operation:he_power_train:electricity:energy_per_fu",
        units="W*h",
        val=energy_per_flight_hours,
    )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00095931, rel=1e-3
    )
    # Slight decrease from 0.00096246 because weirdly enough its more energy efficient to fly
    # seesaws so the less time in cruise the betters ^^'


def test_lca_pipistrel_full_aging_effect_including_econ_degradation_sensitivity():
    """
    Tests that contains:
    - The same as the previous test except we will do it on several value of the max airframe hours
        between 3000h and 5000h instead of a fixed 4000h.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca_proper_aging_with_econ.xml"
    process_file_name = "pipistrel_configuration_with_lca_proper_aging_with_econ.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    for airframe_hours in np.linspace(3000.0, 5000.0, 81):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        # Check that the right input has been set, that is  the number of hours flown before a
        # replacement is required as computed by a separate script
        hours_per_battery = 354.94
        assert problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:hours_per_battery",
            units="h",
        ) == pytest.approx(hours_per_battery, rel=1e-5)
        assert problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_2:hours_per_battery",
            units="h",
        ) == pytest.approx(hours_per_battery, rel=1e-5)

        # This will give us how long into the "life" of the battery
        time_since_last_renewal = airframe_hours % hours_per_battery

        number_of_cycle_since_last_renewal = (
            8.62929561e-04 * time_since_last_renewal**2.0
            + 2.23157428 * time_since_last_renewal
            - 3.09920890
        )
        energy_per_flight_hours = (
            2.32639768e-03 * number_of_cycle_since_last_renewal
            - 5.90621456 * number_of_cycle_since_last_renewal
            + 2.75735887e4
        )
        problem.set_val(
            "data:LCA:operation:he_power_train:electricity:energy_per_fu",
            units="W*h",
            val=energy_per_flight_hours,
        )

        # Run the problem
        problem.run_model()

        # Write the outputs
        file_name = "full_aging_" + str(int(airframe_hours)) + "_airframe_hours_outputs.xml"
        problem.output_file_path = SENSITIVITY_RESULTS_FOLDER_PATH / file_name
        problem.write_outputs()


def test_lca_pipistrel_full_aging_effect_sensitivity():
    """
    Tests that contains:
    - The same as test_lca_pipistrel_full_aging_effect, but on several value of the max airframe
        hours between 3000h and 5000h instead of a fixed 4000h and with the input value which
        correspond to the reference Pipistrel.
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

    # We use a different model to compute the mass per FU for the battery in that test and in the
    # previous one, these equivalent inputs give the same results, they are computed as:
    # hours_per_battery = main_route:duration * battery:lifespan.
    # So the implicit assumption is that the battery capacity doesn't fade with cycle.
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:hours_per_battery",
        units="h",
        val=428.34,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:hours_per_battery",
        units="h",
        val=428.34,
    )
    for airframe_hours in np.linspace(3000.0, 5000.0, 81):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        # Run the problem
        problem.run_model()

        # Write the outputs
        # Write the outputs
        file_name = "reference_" + str(int(airframe_hours)) + "_airframe_hours_outputs.xml"
        problem.output_file_path = SENSITIVITY_RESULTS_FOLDER_PATH / file_name
        problem.write_outputs()
