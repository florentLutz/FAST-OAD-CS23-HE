# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import pytest
import copy

import os.path as pth
import fastoad.api as oad
import logging

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

XML_FILE = "order_assembly.xml"


@pytest.fixture()
def restore_submodels():
    """
    Since the submodels in the configuration file differ from the defaults, this restore process
    ensures subsequent assembly tests run under default conditions.
    """
    old_submodels = copy.deepcopy(oad.RegisterSubmodel.active_models)
    yield
    oad.RegisterSubmodel.active_models = old_submodels


def test_assembly_from_pt_file(restore_submodels):
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
