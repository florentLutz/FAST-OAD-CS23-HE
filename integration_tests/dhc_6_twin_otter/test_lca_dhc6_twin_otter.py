#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2026 ISAE-SUPAERO

import logging
import pathlib
import shutil

import fastoad.api as oad
import pytest

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_lca_twin_otter_pemfc_h2():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lca_pemfc_h2_gas.xml"
    process_file_name = "lca_pemfc_h2_twin_otter.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()
    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        2.557e-05, rel=1e-3
    )


def test_lca_twin_otter_pemfc_h2_recipe():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lca_pemfc_h2_gas.xml"
    process_file_name = "lca_pemfc_h2_twin_otter_recipe.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()
    problem.run_model()
    problem.write_outputs()


def test_lca_twin_otter_for_easn():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lca_twin_otter_for_easn.xml"
    process_file_name = "lca_twin_otter_for_easn.yml"

    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Copy OAD outputs as inputs of the LCA
    shutil.copy(RESULTS_FOLDER_PATH / "oad_process_outputs_ref_for_easn.xml", ref_inputs)

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs, setup, and complete with missing LCA assumptions
    datafile = oad.DataFile(ref_inputs)
    # Great circle distance between Calgary and Toulouse
    datafile.append(
        oad.Variable("data:environmental_impact:delivery:distance", val=8000.0, units="km")
    )
    # Standard assumption
    datafile.append(
        oad.Variable("data:environmental_impact:line_test:duration", val=10.0, units="h")
    )
    # According to TCDS, propeller is aluminium
    datafile.append(
        oad.Variable("data:propulsion:he_power_train:propeller:propeller_1:material", val=0.0)
    )
    datafile.append(
        oad.Variable("data:propulsion:he_power_train:propeller:propeller_2:material", val=0.0)
    )
    datafile.save()

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Based on the wing box lifespan, other structural parts last longer
    problem.set_val("data:TLAR:max_airframe_hours", val=33000, units="h")
    # Based on Viking Air assumptions
    problem.set_val("data:TLAR:flight_hours_per_year", val=1200.0, units="h")
    problem.set_val("data:environmental_impact:buy_to_fly:metallic", val=7.5)

    # Run the problem
    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        1.3027372749944493e-05, rel=1e-2
    )


def test_lca_pemfc_h2_twin_otter_for_easn():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lca_pemfc_h2_twin_otter_for_easn.xml"
    process_file_name = "lca_pemfc_h2_twin_otter_for_easn.yml"

    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Copy OAD outputs as inputs of the LCA
    shutil.copy(
        RESULTS_FOLDER_PATH / "oad_process_outputs_pemfc_h2_gas_retrofit_for_easn.xml", ref_inputs
    )

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs, setup, and complete with missing LCA assumptions
    datafile = oad.DataFile(ref_inputs)
    # Great circle distance between Calgary and Toulouse
    datafile.append(
        oad.Variable("data:environmental_impact:delivery:distance", val=8000.0, units="km")
    )
    # Standard assumption
    datafile.append(
        oad.Variable("data:environmental_impact:line_test:duration", val=10.0, units="h")
    )
    # According to TCDS, propeller is aluminium
    datafile.append(
        oad.Variable("data:propulsion:he_power_train:propeller:propeller_1:material", val=0.0)
    )
    datafile.append(
        oad.Variable("data:propulsion:he_power_train:propeller:propeller_2:material", val=0.0)
    )
    datafile.save()

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Based on the wing box lifespan, other structural parts last longer
    problem.set_val("data:TLAR:max_airframe_hours", val=33000, units="h")
    # Based on Viking Air assumptions
    problem.set_val("data:TLAR:flight_hours_per_year", val=1200.0, units="h")
    problem.set_val("data:environmental_impact:buy_to_fly:metallic", val=7.5)

    # Run the problem
    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        5.44161007e-06, rel=1e-2
    )


def test_lca_pemfc_h2_twin_otter_hybrid_for_easn():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lca_pemfc_h2_twin_otter_hybrid_for_easn.xml"
    process_file_name = "lca_pemfc_h2_twin_otter_hybrid_for_easn.yml"

    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Copy OAD outputs as inputs of the LCA
    shutil.copy(
        RESULTS_FOLDER_PATH / "oad_process_outputs_pemfc_h2_gas_retrofit_hybrid_for_easn.xml",
        ref_inputs,
    )

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs, setup, and complete with missing LCA assumptions
    datafile = oad.DataFile(ref_inputs)
    # Great circle distance between Calgary and Toulouse
    datafile.append(
        oad.Variable("data:environmental_impact:delivery:distance", val=8000.0, units="km")
    )
    # Standard assumption
    datafile.append(
        oad.Variable("data:environmental_impact:line_test:duration", val=10.0, units="h")
    )
    # According to TCDS, propeller is aluminium
    datafile.append(
        oad.Variable("data:propulsion:he_power_train:propeller:propeller_1:material", val=0.0)
    )
    datafile.append(
        oad.Variable("data:propulsion:he_power_train:propeller:propeller_2:material", val=0.0)
    )
    datafile.save()

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Based on the wing box lifespan, other structural parts last longer
    problem.set_val("data:TLAR:max_airframe_hours", val=33000, units="h")
    # Based on Viking Air assumptions
    problem.set_val("data:TLAR:flight_hours_per_year", val=1200.0, units="h")
    problem.set_val("data:environmental_impact:buy_to_fly:metallic", val=7.5)

    # Run the problem
    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        1.11552675e-05, rel=1e-2
    )
