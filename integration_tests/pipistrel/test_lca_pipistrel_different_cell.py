#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib

import logging

import pytest

import numpy as np

import fastoad.api as oad

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_lca"
RESULTS_SENSITIVITY_FOLDER_PATH = pathlib.Path(__file__).parent / "results_sensitivity_lca"


def test_lca_pipistrel_reference_cell():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%, with the reference cell.
    - A LCA evaluation of the sized configuration without considering the effect of the SOH on the
        functional unit. The effect of the performances on the battery aging will however be
        considered without any tuning.
    - The LCA configuration file will be automatically generated.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca.xml"
    process_file_name = "pipistrel_configuration_with_lca_reference_cell.yml"

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

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan", val=700.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:lifespan", val=700.0
    )
    problem.set_val("data:environmental_impact:buy_to_fly:composite", val=1.5)

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW") == pytest.approx(600, rel=1e-3)
    assert problem.get_val("data:weight:aircraft:OWE") == pytest.approx(428, rel=1e-3)
    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00387437, rel=1e-3
    )


def test_lca_pipistrel_reference_cell_sensitivity():
    """
    Tests that contains:
    - A rerun of the LCA of the reference cell but with varying airframe hours value. Nothing will
        change the OAD process, so we will re-use the output file of the previous test as an input
        file for this process
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    process_file_name = "pipistrel_configuration_with_lca_reference_cell_sensitivity.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_reference_cell.xml"

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Shouldn't be needed but just making sure ^^'
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan", val=700.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:lifespan", val=700.0
    )

    for airframe_hours in np.linspace(3000, 5000, 41):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")
        # Run the problem
        problem.run_model()

        # Write the outputs
        file_name = "reference_cell_" + str(int(airframe_hours)) + "_airframe_hours_outputs.xml"
        problem.output_file_path = RESULTS_SENSITIVITY_FOLDER_PATH / file_name
        problem.write_outputs()


def test_lca_pipistrel_high_energy_density_cell():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%, with a higher energy density cell but with
        lower lifetime.
    - A LCA evaluation of the sized configuration without considering the effect of the SOH on the
        functional unit. The effect of the performances on the battery aging will however be
        considered without any tuning.
    - The LCA configuration file will be written separately to represent the change in battery
        production process.
    - Battery production process for Si-NMC battery is not available in EcoInvent, so we will use
        the surrogate process proposed in :cite:`pollet:2023`. This surrogate however assumes that
        for 1 kg of silicon nanowire in the anode, we need 5kg of silicon. For now we will assume a
        1 to 1 ratio, which might change in the future.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca.xml"
    process_file_name = "pipistrel_configuration_with_lca_high_energy_density_cell.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # In addition to a change in battery production process, the characteristics of the battery
    # packs are changed. We will assume same polarization curve and small relative capacity effect
    # . We need at least 4 "fake" points for the relative capacity effect as it assumed to be a
    # deg 3 polynomial
    problem.model_options["*"] = {
        "cell_capacity_ref": 1.34,
        "cell_weight_ref": 11.7e-3,
        "reference_curve_current": [100.0, 1000.0, 3000.0, 5100.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }

    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=550.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=550.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=550.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=550.0)

    # In addition to a change in battery production process, the lifespan of the batteries is also
    # affected, and they need replacement way more often.
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan", val=150.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:lifespan", val=150.0
    )

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=4.0,
        units="h**-1",
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
        val=4.0,
        units="h**-1",
    )
    problem.set_val("data:environmental_impact:buy_to_fly:composite", val=1.5)

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW") == pytest.approx(494, abs=1)
    assert problem.get_val("data:weight:aircraft:OWE") == pytest.approx(322, abs=1)

    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(38.8, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:mass", units="kg"
    ) == pytest.approx(38.8, rel=1e-2)

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00800477, rel=1e-3
    )


def test_lca_pipistrel_high_energy_density_cell_sensitivity():
    """
    Tests that contains:
    - A rerun of the LCA of the high energy density cell but with varying airframe hours value.
        Nothing will change the OAD process, so we will re-use the output file of the previous test
        as an input file for this process
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    process_file_name = "pipistrel_configuration_with_lca_high_energy_density_cell_sensitivity.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_energy_density_cell.xml"

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Shouldn't be needed but just making sure ^^'
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan", val=150.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:lifespan", val=150.0
    )

    for airframe_hours in np.linspace(3000, 5000, 41):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        # Run the problem
        problem.run_model()

        # Write the outputs
        file_name = (
            "high_energy_density_cell_" + str(int(airframe_hours)) + "_airframe_hours_outputs.xml"
        )
        problem.output_file_path = RESULTS_SENSITIVITY_FOLDER_PATH / file_name
        problem.write_outputs()


def test_lca_pipistrel_long_lifespan_cell():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%, with a higher energy density cell but with
        lower lifetime.
    - A LCA evaluation of the sized configuration without considering the effect of the SOH on the
        functional unit. The effect of the performances on the battery aging will however be
        considered without any tuning.
    - The LCA configuration file will be written separately to represent the change in battery
        production process.
    - Battery production process for Skeleton SuperBattery is not available in EcoInvent, so we will
        use a surrogate process inspired from what is done in :cite:`pollet:2023`. This surrogate
        assumes that producing a cathode in graphite is the same as producing an anode in graphite.
    - A small observation to be made is that in the 3.9.1 version of EcoInvent the impact of
        graphite anode production per kg is around 4 times lower than the production of the cathode
        in the reference cell per kg, meaning the impact of cell production will be smaller. In more
        recent version of EcoInvent those two process have the same order of magnitude for the
        impacts so we might be underestimating the impacts of battery production.
    - As is the case in application where the battery mass is very high, the OAD process oscillates
        a lot in the last iteration which actually increases the residuals. To stop them at a
        reasonable value, the tolerance has been decreased.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca.xml"
    process_file_name = "pipistrel_configuration_with_lca_high_lifespan_cell.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # In addition to a change in battery production process, the characteristics of the battery
    # packs are changed. We will assume same polarization curve and small relative capacity effect
    # . We need at least 4 "fake" points for the relative capacity effect as it assumed to be a
    # deg 3 polynomial.
    # The code seems to have a hard time converging with "big" cells, so instead of having a few
    # "big" cells we'll have more smaller one.
    # This cell actually doesn't correspond to the battery we aimed at reproducing
    # (https://www.skeletontech.com/superbattery) for the sole reason that the mass diverges. This
    # is actually the closest we can get without divergence so we are very optimistic on the
    # battery energy density but even then the single is greater than the reference case.
    # We tke data from the D60 cell
    problem.model_options["*"] = {
        "cell_capacity_ref": 3.3,
        "cell_weight_ref": 0.09
        / 1.1363636363636365,  # 0.09 Corresponds to the reference conditions for the D60
        "reference_curve_current": [100.0, 10000.0, 30000.0, 46000.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
        "cut_off_voltage": 1.0,
        "cut_off_propeller_efficiency": 0.75,
    }

    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1500.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=1300.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=1500.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=1500.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=1500.0)

    # In addition to a change in battery production process, the lifespan of the batteries is also
    # affected, and they need replacement way less often.
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan", val=50000.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:lifespan", val=50000.0
    )

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=20.0,
        units="h**-1",
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
        val=20.0,
        units="h**-1",
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:module:number_cells", val=150.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:module:number_cells", val=150.0
    )

    # To aid convergence
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=20.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=20.0
    )
    problem.set_val("data:environmental_impact:buy_to_fly:composite", val=1.5)

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW") == pytest.approx(2463, abs=1)
    assert problem.get_val("data:weight:aircraft:OWE") == pytest.approx(2291, abs=1)

    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(662.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:mass", units="kg"
    ) == pytest.approx(662.0, rel=1e-2)

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.0046992541767277955, rel=1e-3
    )


def test_lca_pipistrel_long_lifespan_cell_sensitivity():
    """
    Tests that contains:
    - A rerun of the LCA of the high lifespan cell but with varying airframe hours value.
        Nothing will change the OAD process, so we will re-use the output file of the previous test
        as an input file for this process
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    process_file_name = "pipistrel_configuration_with_lca_high_lifespan_cell_sensitivity.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_lifespan_cell.xml"

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Shouldn't be needed but just making sure ^^'
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan", val=50000.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:lifespan", val=50000.0
    )

    for airframe_hours in np.linspace(3000, 5000, 41):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")
        # Run the problem
        problem.run_model()

        # Write the outputs
        file_name = "high_lifespan_cell_" + str(int(airframe_hours)) + "_airframe_hours_outputs.xml"
        problem.output_file_path = RESULTS_SENSITIVITY_FOLDER_PATH / file_name
        problem.write_outputs()
