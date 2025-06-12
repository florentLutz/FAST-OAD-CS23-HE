# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth
import fastoad.api as oad
import logging

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
OUT_FOLDER_PATH = pth.join(pth.dirname(__file__), "outputs")

XML_FILE = "order_assembly.xml"
NB_POINTS_TEST = 50
COEFF_DIFF = 0.0


def test_assembly_from_pt_file():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    process_file_name = "mission_config_order_problem.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, XML_FILE)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    # Run with another order
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    process_file_name = "mission_config_order_correct.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, XML_FILE)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()
    problem.run_model()
