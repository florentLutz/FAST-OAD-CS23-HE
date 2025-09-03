#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib
import logging

import pytest

import fastoad.api as oad

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_sizing_cessna_208b():
    """
    We will not do the latest version of the C208 rather we'll do the C208 before the re-engining so
    it will be rated at 675hp.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_c208.xml"
    process_file_name = "full_sizing_c208.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_cirrus.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        982.00, rel=5e-2
    )
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        2146.0, rel=5e-2
    )  # From the POH found online + weight of seats at around 100kg
    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3968.0, rel=5e-2
    )


def test_op_mission_cessna_208b():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_c208_op_mission.xml"
    process_file_name = "op_mission_fuel.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()


def test_sizing_hybrid_cessna_208b():
    """
    Parallel hybrid version of the c208b, in this first version we won't allow resizing of the
    turboshaft, we'll just "replace" thermal power by electric power. Designed for 200nm. Max
    payload will be set equal to payload of the orig C208B on its operationnal mission to not give
    the hybrid an unwanted advantage linked with the resizing
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_c208_retrofit.xml"
    process_file_name = "hybrid_c208_retrofit.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}
    problem.model_options["*"] = {
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

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3968.0, rel=5e-2
    )
