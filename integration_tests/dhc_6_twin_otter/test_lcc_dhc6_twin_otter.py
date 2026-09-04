#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2026 ISAE-SUPAERO

import logging
import pathlib
import shutil

import numpy as np
import openmdao.api as om
import fastoad.api as oad
import pytest

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_lcc_twin_otter_for_easn():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lcc_twin_otter_for_easn.xml"
    process_file_name = "lcc_twin_otter_for_easn.yml"

    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Run the problem
    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:cost:operation:revenue_per_rpk", units="USD/km") == pytest.approx(
        0.7615, rel=1e-2
    )


def test_lcc_pemfc_h2_twin_otter_for_easn():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lcc_pemfc_h2_twin_otter_for_easn.xml"
    process_file_name = "lcc_pemfc_h2_twin_otter_for_easn.yml"

    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Run the problem
    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:cost:operation:revenue_per_rpk", units="USD/km") == pytest.approx(
        1.3288, rel=1e-2
    )


def test_lcc_pemfc_h2_twin_otter_hybrid_for_easn():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lcc_pemfc_h2_twin_otter_hybrid_for_easn.xml"
    process_file_name = "lcc_pemfc_h2_twin_otter_hybrid_for_easn.yml"

    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Run the problem
    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:cost:operation:revenue_per_rpk", units="USD/km") == pytest.approx(
        0.708, rel=1e-2
    )


def test_lcc_doe_pemfc_h2_twin_otter_hybrid_for_easn():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_retrofit_pemfc_h2_twin_otter_hybrid_optim_for_easn.xml"
    process_file_name = "retrofit_pemfc_h2_twin_otter_hybrid_optim_lcc_for_easn.yml"

    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    problem.model_options["*propeller_*"] = {"mass_as_input": True}
    problem.model_options["*pemfc_stack_*"] = {"mass_from_specific_power": True}

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Any lower than 60% gives unrealistic values for turboshaft.
    for power_split in [60, 65, 70, 75, 80]:
        problem.set_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
            val=power_split,
            units="percent",
        )
        problem.set_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_2:power_split",
            val=power_split,
            units="percent",
        )
        # Run the problem
        problem.output_file_path = (
            RESULTS_FOLDER_PATH / "percent_split_doe_lcc" / (str(power_split) + ".xml")
        )
        problem.run_model()
        problem.write_outputs()
