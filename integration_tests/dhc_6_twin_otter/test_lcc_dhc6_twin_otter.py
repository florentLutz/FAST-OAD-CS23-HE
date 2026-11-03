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


def test_lcc_twin_otter():
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

    assert problem.get_val("data:cost:msp_per_unit", units="USD") == pytest.approx(
        5.403e6, rel=1e-2
    )
    # 8.4% short from the 2024 Q2 listed price 5.9M USD
    assert problem.get_val("data:cost:operation:revenue_per_rpk", units="USD/km") == pytest.approx(
        0.766, rel=1e-2
    )

    assert problem.get_val("data:cost:production:profitability_index")[-1] == pytest.approx(
        1.693, rel=1e-2
    )
    assert problem.get_val("data:cost:operation:profitability_index")[-1] == pytest.approx(
        1.362, rel=1e-2
    )


def test_lcc_pemfc_h2_twin_otter_with_baseline_ticket_price():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lcc_pemfc_h2_twin_otter.xml"
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

    assert problem.get_val("data:cost:msp_per_unit", units="USD") == pytest.approx(4.52e6, rel=1e-2)
    assert problem.get_val("data:cost:production:profitability_index")[-1] == pytest.approx(
        1.535, rel=1e-2
    )
    assert problem.get_val("data:cost:operation:profitability_index")[-1] == pytest.approx(
        -1.131, rel=1e-2
    )


def test_lcc_pemfc_h2_twin_otter_hybrid_with_baseline_ticket_price():
    """
    Calculate the revenue per passenger-kilometer of a hybrid model with defined annual profit
    margin.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "lcc_pemfc_h2_twin_otter_hybrid.xml"
    process_file_name = "lcc_pemfc_h2_twin_otter_hybrid.yml"

    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Run the problem
    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:cost:msp_per_unit", units="USD") == pytest.approx(5.07e6, rel=1e-2)
    assert problem.get_val("data:cost:production:profitability_index")[-1] == pytest.approx(
        1.6407, rel=1e-2
    )
    assert problem.get_val("data:cost:operation:profitability_index")[-1] == pytest.approx(
        0.293, rel=1e-2
    )
