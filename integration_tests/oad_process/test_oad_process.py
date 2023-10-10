import os
import os.path as pth
from shutil import rmtree
import logging

import numpy as np

import pytest

import fastoad.api as oad
import openmdao.api as om

import plotly.graph_objects as go

from fastga_he.gui.power_train_network_viewer import power_train_network_viewer

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


@pytest.fixture(scope="module")
def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def test_fuel_and_battery(cleanup):

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "full_sizing_fuel_and_battery.xml"
    process_file_name = "full_sizing_fuel_and_battery.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1200.0)
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=25.0
    )
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        761.0, rel=1e-2
    )
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(24.71, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(128.0, rel=1e-2)


def test_sizing_sr22(cleanup):

    # TODO: Recheck inputs

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22.xml"
    process_file_name = "full_sizing_fuel.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem, show_browser=True)

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1000.0)
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        1747.0, rel=1e-2
    )
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        275.76, rel=1e-2
    )


def test_sizing_fuel_and_battery_share(cleanup):

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "full_sizing_fuel_and_battery_share.xml"
    process_file_name = "full_sizing_fuel_and_battery_share.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem, show_browser=True)

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=700.0)
    problem.set_val("data:geometry:wing:area", units="m**2", val=10.0)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(23.38, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(32.633, abs=1e-2)
    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        815.414, rel=1e-2
    )


def test_dep_aircraft(cleanup):

    pt_file_path = pth.join(DATA_FOLDER_PATH, "dep_assembly.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "dep_assembly.html")

    if not os.path.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "full_sizing_dep_ac.xml"
    process_file_name = "full_sizing_dep_ac.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1400.0)
    problem.set_val("data:geometry:wing:area", units="m**2", val=12.0)
    problem.set_val("data:propulsion:he_power_train:mass", units="kg", val=955.0)

    model = problem.model
    recorder = om.SqliteRecorder(pth.join(DATA_FOLDER_PATH, "cases.sql"))
    solver = model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True
    solver.recording_options["record_outputs"] = True

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        1422.0, rel=1e-2
    )


def test_read_case_recorder():

    recorder_data_file_path = pth.join(DATA_FOLDER_PATH, "cases_oscillate.sql")

    cr = om.CaseReader(recorder_data_file_path)

    mtow_array = []
    wing_area_array = []
    pt_weight_array = []
    airframe_weight = []
    battery_pack_1_mass = []
    battery_pack_1_modules_nb = []
    battery_pack_1_capa_mult = []
    battery_pack_1_c_rate_mult = []
    battery_pack_1_soc_mission = []
    battery_pack_2_mass = []

    for i in range(1, 49):

        case = cr.get_case("rank0:root._solve_nonlinear|0|NonlinearBlockGS|" + str(i))

        mtow_array.append(float(case.outputs["data:weight:aircraft:MTOW"]))
        wing_area_array.append(float(case.outputs["data:geometry:wing:area"]))
        pt_weight_array.append(float(case.outputs["data:propulsion:he_power_train:mass"]))
        airframe_weight.append(float(case.outputs["data:weight:airframe:mass"]))
        battery_pack_1_mass.append(
            float(case.outputs["data:propulsion:he_power_train:battery_pack:battery_pack_1:mass"])
        )
        battery_pack_1_modules_nb.append(
            float(
                case.outputs[
                    "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules"
                ]
            )
        )
        battery_pack_1_c_rate_mult.append(
            float(
                case.outputs[
                    "data:propulsion:he_power_train:battery_pack:battery_pack_1:c_rate_multiplier"
                ]
            )
        )
        battery_pack_1_capa_mult.append(
            float(
                case.outputs[
                    "data:propulsion:he_power_train:battery_pack:battery_pack_1:capacity_multiplier"
                ]
            )
        )
        battery_pack_1_soc_mission.append(
            float(
                case.outputs["data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min"]
            )
        )
        battery_pack_2_mass.append(
            float(case.outputs["data:propulsion:he_power_train:battery_pack:battery_pack_2:mass"])
        )

    fig = go.Figure()
    multiplicative_factor = np.maximum(battery_pack_1_capa_mult, battery_pack_1_c_rate_mult)
    multiplicative_factor = np.clip(multiplicative_factor, 0.9, 1.0 / 0.9)

    scatter_number_of_battery_module = go.Scatter(
        x=np.arange(1, 50),
        y=battery_pack_1_modules_nb,
        mode="lines+markers",
        name="Number of battery modules",
    )
    fig.add_trace(scatter_number_of_battery_module)
    fig.update_layout(
        title_text="Evolution of the number of battery module during the sizing process",
        title_x=0.5,
        xaxis_title="Number of modules [-]",
        yaxis_title="Number of iteration [-]",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    # scatter_multiplicative_factor = go.Scatter(
    #     x=np.arange(1, 50),
    #     y=multiplicative_factor,
    #     mode="lines+markers",
    #     name="Multiplicative factor on the number of modules",
    # )
    # fig.add_trace(scatter_multiplicative_factor)
    # fig.update_layout(
    #     title_text="Evolution of the multiplicative factor on the number of modules",
    #     title_x=0.5,
    #     xaxis_title="Multiplicative factor [-]",
    #     yaxis_title="Number of iteration [-]",
    #     legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    # )

    fig.show()

    # plt.plot(mtow_array)
    # plt.plot(wing_area_array)
    # plt.plot(np.array(airframe_weight) / airframe_weight[0])
    # plt.plot(np.array(pt_weight_array) / pt_weight_array[0])
    # plt.plot(battery_pack_1_mass)
    # plt.plot(battery_pack_2_mass)
    # plt.plot(battery_pack_1_modules_nb)
    # plt.plot(battery_pack_1_c_rate_mult, label="C_rate")
    # plt.plot(battery_pack_1_capa_mult, label="Capacity")
    # plt.plot(battery_pack_1_soc_mission, label="SOC min")
    # plt.legend()
    # plt.show()
