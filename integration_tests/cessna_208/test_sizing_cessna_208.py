#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib
import logging

import numpy as np

import pytest

import fastoad.api as oad

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
DOE_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_doe_no_resizing"
DOE_RESIZED_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_doe_resizing"


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
    problem.model_options["*turboshaft_1*"] = {
        "adjust_sfc": True,
        "reference_rated_power": [300, 503.3475],
        "reference_k_sfc": [1.2, 1.05],
    }
    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.setup()

    problem.set_val(
        "data:weight:aircraft:MTOW",
        units="kg",
        val=3968.0,
    )
    problem.model.aircraft_sizing.nonlinear_solver.options["maxiter"] = 30

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
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(434.0, rel=1e-2)


def test_doe_sizing_hybrid_cessna_208b():
    """
    Same test as above except with varying values of the power share parameter.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_c208_retrofit.xml"
    process_file_name = "hybrid_c208_retrofit.yml"

    # We'll go from 10 kW below the max power to 10 kW below the cruise power (from experience it is
    # going to diverge at this point)
    power_shares = np.arange(270.0, 391.0, 10.0)

    for power_share in power_shares:
        configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
        problem = configurator.get_problem()

        # Create inputs
        ref_inputs = DATA_FOLDER_PATH / xml_file_name

        problem.write_needed_inputs(ref_inputs)
        problem.read_inputs()

        problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}
        problem.model_options["*turboshaft_1*"] = {
            "adjust_sfc": True,
            "reference_rated_power": [300, 503.3475],
            "reference_k_sfc": [1.2, 1.05],
        }
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

        problem.set_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_share",
            units="kW",
            val=power_share,
        )
        problem.set_val(
            "data:weight:aircraft:MTOW",
            units="kg",
            val=3968.0,
        )

        problem.run_model()

        problem.output_file_path = DOE_RESULTS_FOLDER_PATH / (
            str(int(round(power_share, 0))) + "_power_share.xml"
        )
        problem.write_outputs()


def test_sizing_hybrid_cessna_208b_better_fit():
    """
    Same as above, except the turboshaft is resized up to the same turboshaft as the Kodiak.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_c208_retrofit.xml"
    process_file_name = "hybrid_c208_retrofit_perfect_fit.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}
    problem.model_options["*turboshaft_1*"] = {
        "adjust_sfc": True,
        "reference_rated_power": [300, 503.3475],
        "reference_k_sfc": [1.2, 1.05],
    }
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
    problem.set_val(
        "data:weight:aircraft:MTOW",
        units="kg",
        val=3968.0,
    )
    # This was found to be the best point
    # problem.set_val(
    #     "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_share",
    #     units="kW",
    #     val=277.0,
    # )

    # We won't fix the turboshaft here, rather let the code resize it for a perfect fit. We will
    # assume that the turboshaft will be of the same family as the original one. So the same thermo-
    # dynamic parameter will be kept, same k_sfc as well.

    problem.run_model()

    problem.output_file_path = RESULTS_FOLDER_PATH / "oad_process_outputs_he_better_fit.xml"
    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3968.0, rel=5e-2
    )
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(332.0, rel=1e-2)


def test_doe_sizing_hybrid_cessna_208b_better_fit():
    """
    Same test as above except with varying values of the power share parameter.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_c208_retrofit.xml"
    process_file_name = "hybrid_c208_retrofit_perfect_fit.yml"

    # We'll go from 10 kW below the max power to 10 kW below the cruise power (from experience it is
    # going to diverge at this point)
    power_shares = np.arange(270.0, 401.0, 10.0)
    # power_shares = np.arange(270.0, 290.5, 1.0)

    for power_share in power_shares:
        configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
        problem = configurator.get_problem()

        # Create inputs
        ref_inputs = DATA_FOLDER_PATH / xml_file_name

        problem.write_needed_inputs(ref_inputs)
        problem.read_inputs()

        problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}
        problem.model_options["*turboshaft_1*"] = {
            "adjust_sfc": True,
            "reference_rated_power": [300, 503.3475],
            "reference_k_sfc": [1.2, 1.05],
        }
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

        problem.set_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_share",
            units="kW",
            val=power_share,
        )

        problem.set_val(
            "data:weight:aircraft:MTOW",
            units="kg",
            val=3968.0,
        )

        problem.run_model()

        problem.output_file_path = DOE_RESIZED_RESULTS_FOLDER_PATH / (
            str(int(round(power_share, 0))) + "_power_share.xml"
        )
        problem.write_outputs()


def test_resizing_c208b_new_mission():
    """
    Sizes a pseudo C208 with the same mission as the one the hybrid version was sized on, with  
    the turboshaft resized.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_c208_with_resizing.xml"
    process_file_name = "full_sizing_c208_with_resizing.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs and run the problem
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()
    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        400.00, rel=5e-2
    )  # Should be lower than the fuel of the original on the off design mission (428 kg)
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        2028.0, rel=5e-2
    )  # Should be lower than the original OWE 2146
    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3578.0, rel=5e-2
    )  # Should be lower than the TOW of the original on the off design mission (3708 kg)


def test_sizing_hybrid_cessna_208b_better_fit_full_sizing():
    """
    Full sizing of the hybrid Cessna 208
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_c208.xml"
    process_file_name = "hybrid_c208_full_sizing.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*motor_1*"] = {"adjust_rpm_rating": True}
    problem.model_options["*turboshaft_1*"] = {
        "adjust_sfc": True,
        "reference_rated_power": [300, 503.3475],
        "reference_k_sfc": [1.2, 1.05],
    }
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
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min",
        val=75.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:c_rate_max",
        val=3.0,
        units="h**-1",
    )
    problem.set_val(
        "data:weight:aircraft:MTOW",
        units="kg",
        val=3968.0,
    )
    problem.set_val(
        "data:geometry:wing:area",
        units="m**2",
        val=25.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:current_in_max",
        val=100.0,
        units="A",
    )
    problem.set_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_max",
        val=400.0,
        units="kW",
    )

    # We won't fix the turboshaft here, rather let the code resize it for a perfect fit. We will
    # assume that the turboshaft will be of the same family as the original one. So the same thermo-
    # dynamic parameter will be kept, same k_sfc as well.

    problem.run_model()
    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3968.0, rel=5e-2
    )
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(332.0, rel=1e-2)
