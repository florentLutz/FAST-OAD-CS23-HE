# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import pathlib
from shutil import rmtree
import logging

import pytest

import fastoad.api as oad

from utils.filter_residuals import filter_residuals
from uqpce.pce.pce import PCE

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
DOE_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_doe"
DOE_RESULTS_FOLDER_PATH_SPLIT = pathlib.Path(__file__).parent / "results_doe_power_split"
WORKDIR_FOLDER_PATH = pathlib.Path(__file__).parent / "workdir"


def test_sizing_sr22_uqpce():
    samp_count = 2
    aleat_cnt = 500000
    epist_cnt = 1

    pce = PCE(
        order=2, verbose=True, outputs=False, plot=False, aleat_samp_size=aleat_cnt,
        epist_samp_size=epist_cnt
    )

    # Add two normal variables
    pce.add_variable(distribution='normal', mean=850, stdev=30, name='data:TLAR:range') # NM

    # Generate samples that correspond to the input variables
    Xt = pce.sample(count=samp_count)
    Yt = []
    for x in Xt:
        """Test the overall aircraft design process with wing positioning under VLM method."""
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
        logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

        # Define used files depending on options
        xml_file_name = "input_sr22.xml"
        process_file_name = "full_sizing_fuel.yml"

        configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
        problem = configurator.get_problem()

        ref_inputs = DATA_FOLDER_PATH / xml_file_name

        problem.write_needed_inputs(ref_inputs)
        problem.read_inputs()
        problem.setup()

        problem.set_val(
            "data:TLAR:range",
            val=x,
            units="NM",
        )

        problem.run_model()

        _, _, residuals = problem.model.get_nonlinear_vectors()
        residuals = filter_residuals(residuals)

        problem.write_outputs()

        Yt.append(problem.get_val("data:weight:aircraft:MTOW", units="kg"))


    print(Xt)
    print(Yt)



# def test_sizing_sr22():
#     # TODO: Check why he needs the propulsion data as inputs
#     # ANS: still used for the Z_cg of the aircraft which is assumed to have only a minor influence
#
#     """Test the overall aircraft design process with wing positioning under VLM method."""
#     logging.basicConfig(level=logging.WARNING)
#     logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
#     logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
#
#     # Define used files depending on options
#     xml_file_name = "input_sr22.xml"
#     process_file_name = "full_sizing_fuel.yml"
#
#     configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
#     problem = configurator.get_problem()
#
#     # Create inputs
#     ref_inputs = DATA_FOLDER_PATH / xml_file_name
#     # n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_cirrus.html")
#     # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
#
#     problem.write_needed_inputs(ref_inputs)
#     problem.read_inputs()
#     problem.setup()
#
#     # om.n2(problem, show_browser=False, outfile=n2_path)
#
#     problem.run_model()
#
#     _, _, residuals = problem.model.get_nonlinear_vectors()
#     residuals = filter_residuals(residuals)
#
#     problem.write_outputs()
#
#     assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
#         1601.0, rel=1e-2
#     )
#     assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(992.0, rel=1e-2)
#     assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
#         234.00, rel=1e-2
#     )
