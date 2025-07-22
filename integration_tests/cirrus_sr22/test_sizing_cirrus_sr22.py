# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pathlib
from shutil import rmtree
import logging

import pytest

import fastoad.api as oad

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
WORKDIR_FOLDER_PATH = pathlib.Path(__file__).parent / "workdir"


@pytest.fixture(scope="module")
def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree(WORKDIR_FOLDER_PATH, ignore_errors=True)


def test_sizing_sr22():
    # TODO: Check why he needs the propulsion data as inputs
    # ANS: still used for the Z_cg of the aircraft which is assumed to have only a minor influence

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22.xml"
    process_file_name = "full_sizing_fuel.yml"

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

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        1601.0, rel=1e-2
    )
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(992.0, rel=1e-2)
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        234.00, rel=1e-2
    )


def test_sizing_sr22_electric_original():
    """
    Tests an electric sr22 with the same climb, cruise, descent and reserve profile as the original
    one but a range of 100 nm. The only change is the ratio between stall speed and reserve speed
    which is set at 1.3 (ratio between approach and stall)
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_electric.xml"
    process_file_name = "full_sizing_electric.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()


def test_sizing_sr22_electric_two_motors():
    """
    Same tests as above except, to increase confidence in e-motor results we will take two "stacked"
    motors instead of a big one, as is doable with EMRAX engine. They will be coupled by a gearbox
    of 0 kg and 0.99 efficiency. Also, because the E-motor doesn't lose power with altitude, the
    climb rate at cruise level will be improved which should result in better energy efficiency.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_electric_two_motors.xml"
    process_file_name = "full_sizing_electric_two_motors.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3172.0, rel=1e-2
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(745.0, abs=1e-2)


def test_sizing_sr22_electric_two_motors_improved_range_sensitivity():
    """
    Same tests as above except we test some possible improvements:
    - Not certify the aircraft under IFR but rather VFR, in our context it means a reserve of 30
        min instead of 45 min
    - Keep decreasing the range to 50 nm (only 23% of flights) (not worth not done) but instead
        we'll look at the evolution of MTOW and battery mass with range
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_electric_two_motors.xml"
    process_file_name = "full_sizing_electric_two_motors.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val("data:mission:sizing:main_route:reserve:duration", units="min", val=30.0)

    # Baseline
    problem.set_val("data:TLAR:range", units="NM", val=100.0)
    # problem.set_val("data:TLAR:range", units="NM", val=175.0)

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.output_file_path = RESULTS_FOLDER_PATH / "full_sizing_elec_out_two_motors_improved.xml"
    problem.write_outputs()

    # Range tested
    # 50, 75, 100, 125, 150, 175

    # Corresponding MTOW [kg]
    # 2092.0, 2380, 2763, 3210, 3882, 5368

    # Corresponding battery mass [kg]
    # 2.0 * 382, 2.0 * 479, 2.0 * 605, 2.0 * 757, 2.0 * 981, 2.0 * 1454


def test_sizing_sr22_electric_two_motors_improved_plus_range_sensitivity():
    """
    Same tests as above except we replace the cell with a high energy density cell. In that case we
    will take a cell from the company Amprius with a gravimetric energy density of 395 Wh/kg,
    whereas before it was 261.4 Wh/kg
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22_electric_two_motors.xml"
    process_file_name = "full_sizing_electric_two_motors.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 1.34,
        "cell_weight_ref": 11.7e-3,
        "reference_curve_current": [100.0, 1000.0, 3000.0, 5100.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }

    problem.setup()

    problem.set_val("data:mission:sizing:main_route:reserve:duration", units="min", val=30.0)

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

    # Baseline
    problem.set_val("data:TLAR:range", units="NM", val=100.0)
    # problem.set_val("data:TLAR:range", units="NM", val=360.0)

    # om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "full_sizing_elec_out_two_motors_plus_improved.xml"
    )
    problem.write_outputs()

    # Range tested
    # 50, 75, 100, 125, 150, 175, 200, 225, 251, 275, 280, 290, 305, 315, 323, 333, 343, 353, 360

    # Corresponding MTOW [kg] 1591, 1726, 1867, 2008, 2160, 2327, 2509, 2721, 2982, 3243, 3304,
    # 3435, 3654, 3819, 3965, 4172, 4419, 4733, 4989

    # Corresponding battery mass [kg] 2.0 * 209, 2.0 * 257, 2.0 * 306, 2.0 * 355, 2.0*408,
    # 2.0 * 465, 2.0 * 527, 2.0 * 597, 681, 2.0*770, 2.0 * 790, 2.0 * 834, 2.0 * 907, 2.0 * 961,
    # 2.0 * 1009, 2.0 * 1077, 2.0 * 1156, 2.0 * 1256, 2.0 * 1337
