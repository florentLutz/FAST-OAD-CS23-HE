import time
import os
import os.path as pth
import logging

import pytest

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import fastoad.api as oad
import openmdao.api as om

from PIL import Image

from fastga.utils.postprocessing.analysis_and_plots import mass_breakdown_bar_plot

from fastga_he.gui.power_train_network_viewer import power_train_network_viewer
from fastga_he.gui.analysis_and_plots import (
    aircraft_geometry_plot,
)
from fastga_he.gui.performances_viewer import MARKER_DICTIONARY, COLOR_DICTIONARY

from fastga_he.powertrain_builder.powertrain import PROMOTION_FROM_MISSION

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
RESULTS_SENSITIVITY_FOLDER_PATH = pth.join(pth.dirname(__file__), "results_sensitivity")


COLORS = px.colors.qualitative.Prism


def test_pipistrel_velis_electro():
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source.xml"
    process_file_name = "pipistrel_configuration.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Removing previous case and adding a recorder
    recorder_path = pth.join(RESULTS_FOLDER_PATH, "pipistrel_cases.sql")

    if pth.exists(recorder_path):
        os.remove(recorder_path)

    recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # Run the problem
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        600.00, rel=1e-2
    )
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(24.98, abs=1e-2)


def test_pipistrel_performances_with_lower_soh():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_inputs_performances_lower_SOH.xml"
    process_file_name = "pipistrel_performances_lower_soh.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    state_of_health = 80.0
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:state_of_health",
        val=state_of_health,
        units="percent",
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:state_of_health",
        val=state_of_health,
        units="percent",
    )

    problem.model.performances.nonlinear_solver.options["rtol"] = 1e-8
    problem.model.performances.solve_equilibrium.compute_dep_equilibrium.nonlinear_solver.options[
        "iprint"
    ] = 2

    problem.run_model()

    problem.write_outputs()

    duration_main_route = (
        problem.get_val("data:mission:operational:duration", units="min")
        - problem.get_val("data:mission:operational:reserve:duration", units="min")
        - problem.get_val("data:mission:operational:taxi_in:duration", units="min")
        - problem.get_val("data:mission:operational:taxi_out:duration", units="min")
    )
    assert duration_main_route == pytest.approx(15.86, abs=1.0)


def test_pipistrel_velis_electro_lower_soc_start_mission():
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_lower_soc_start.xml"
    process_file_name = "pipistrel_lower_soc_start_configuration.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start",
        units="percent",
        val=80.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:SOC_mission_start",
        units="percent",
        val=80.0,
    )

    # Run the problem
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.output_file_path = pth.join(RESULTS_FOLDER_PATH, "pipistrel_lower_soc_start_out.xml")
    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        650.00, rel=1e-2
    )
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(25.94, abs=1e-2)
    # vs 25 kWh with the classic aircraft, so the increase is due to the increased weight and not
    # the decreased SOC start


def test_pipistrel_velis_electro_with_lca_default_battery_lifespan():
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca.xml"
    process_file_name = "pipistrel_configuration_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # For quick test
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:aging:calendar_effect_k_factor",
        val=0.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:aging:calendar_effect_k_factor",
        val=0.0,
    )

    # Run the problem
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        580.00, rel=1e-2
    )
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(24.74, abs=1e-2)


def test_pipistrel_velis_electro_with_lca_varying_start_soc():
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca.xml"
    process_file_name = "pipistrel_configuration_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # Run the problem
    problem.run_model()

    # Now we change the SOC start a little
    soc_start_array = np.linspace(100.0, 80.0, 10)
    for soc_start in soc_start_array:
        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start",
            units="percent",
            val=soc_start,
        )
        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_2:SOC_mission_start",
            units="percent",
            val=soc_start,
        )

        problem.run_model()

        problem.output_file_path = pth.join(
            RESULTS_SENSITIVITY_FOLDER_PATH, str(int(soc_start)) + "_soc_start_out.xml"
        )
        problem.write_outputs()


def test_post_process_results():
    fig = go.Figure()

    socs_start_mission = []
    single_scores = []
    battery_mass_per_fu = []
    battery_mass = []

    generic_list = []

    for file in os.listdir(RESULTS_SENSITIVITY_FOLDER_PATH):
        datafile = oad.DataFile(pth.join(RESULTS_SENSITIVITY_FOLDER_PATH, file))
        socs_start_mission.append(
            datafile[
                "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start"
            ].value[0]
        )
        single_scores.append(datafile["data:environmental_impact:single_score"].value[0])
        battery_mass_per_fu.append(
            datafile[
                "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass_per_fu"
            ].value[0]
        )
        battery_mass.append(
            datafile["data:propulsion:he_power_train:battery_pack:battery_pack_1:mass"].value[0]
        )
        generic_list.append(
            datafile["data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan"].value[0]
        )

    sorted_single_scores = [single_scores[i] for i in np.argsort(socs_start_mission)]
    sorted_battery_mass_per_fu = [battery_mass_per_fu[i] for i in np.argsort(socs_start_mission)]
    sorted_socs_start = np.sort(socs_start_mission)

    sorted_generic_list = [generic_list[i] for i in np.argsort(socs_start_mission)]
    # sorted_battery_mass = [battery_mass[i] for i in np.argsort(socs_start_mission)]

    scatter = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_single_scores) / sorted_single_scores[-1],
        mode="lines+markers",
        name="Single score variation",
    )
    fig.add_trace(scatter)
    scatter_2 = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_battery_mass_per_fu) / sorted_battery_mass_per_fu[-1],
        mode="lines+markers",
        name="Battery mass per FU variation",
    )
    fig.add_trace(scatter_2)
    scatter_3 = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_generic_list) / sorted_generic_list[-1],
        mode="lines+markers",
        name="Battery lifespan variation",
    )
    fig.add_trace(scatter_3)

    fig.update_layout(
        title_text="Evolution of single score",
        title_x=0.5,
        xaxis_title="Initial SoC [%]",
        height=800,
        width=1600,
        font_size=18,
    )
    fig.update_yaxes(title="Variation with respect to 100% SOC case")

    fig.show()


def test_pipistrel_heavy_velis_electro():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_heavy_source.xml"
    process_file_name = "pipistrel_heavy_configuration.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # Run the problem
    problem.run_model()

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        661.00, rel=1e-2
    )
    assert problem.get_val("data:mission:sizing:energy", units="kW*h") == pytest.approx(
        26.13, abs=1e-2
    )


def test_pipistrel_detailed_mission():
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_mission_in.xml"
    process_file_name = "pipistrel_mission_configuration.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # Run the problem
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(24.86, abs=1e-2)


def test_pipistrel_op_mission():
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_op_mission_in.xml"
    process_file_name = "pipistrel_op_mission_configuration.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # Run the problem
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    sizing_energy = problem.get_val("data:mission:operational:energy", units="kW*h")
    assert sizing_energy == pytest.approx(20.77, abs=1e-2)


def test_pipistrel_not_detailed_mission():
    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_mission_in.xml"
    process_file_name = "pipistrel_simple_mission_configuration.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    problem.set_val(
        "data:mission:sizing:main_route:descent:descent_rate",
        units="ft/min",
        val=-500.0,
    )
    problem.set_val(
        "convergence:propulsion:he_power_train:propeller:propeller_1:min_power",
        units="W",
        val=500.0,
    )

    # Run the problem
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(24.93, abs=1e-2)


def test_residuals_analyzer():
    # Does not bring much info since the bloody reluctance is so high ...

    cr = om.CaseReader("results/pipistrel_cases.sql")

    solver_case = cr.get_cases("root.nonlinear_solver")
    for case in solver_case:
        max_residual_value = 0.0
        max_residual_name = ""

        for residual in case.residuals:
            # Because those are matrix and I don't want to deal with it
            if "aerodynamics:propeller" not in residual:
                residual_value = sum(np.square(case.residuals[residual]))

                if residual_value > max_residual_value:
                    max_residual_value = residual_value
                    max_residual_name = residual

        # print the result
        print(f"The top residuals is {max_residual_name} with a score of {max_residual_value}.")


def test_residuals_plotter():
    fig = go.Figure()

    cr = om.CaseReader("results/pipistrel_cases.sql")
    solver_case = cr.get_cases("root.nonlinear_solver")

    data_to_plot = []

    for case in solver_case:
        data_to_plot.append(
            float(
                case.outputs[
                    "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules"
                ]
            )
        )

    scatter = go.Scatter(
        y=data_to_plot,
        mode="lines+markers",
    )
    fig.add_trace(scatter)
    fig.update_layout(
        title_text="Evolution of value",
        title_x=0.5,
    )

    fig.show()


def test_comparison_with_data():
    fig = go.Figure()
    datafile = oad.DataFile("results/pipistrel_out.xml")

    # Each battery pack weighs about 70 kg
    # (https://www.pipistrel-aircraft.com/products/velis-electro/#1680814658914-2692ce13-a72a)
    scatter_compare_mass(
        actual_mass=70.0,
        computed_mass=datafile[
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass"
        ].value[0],
        axes_name="Battery pack",
        graph_number=1,
        fig=fig,
    )
    # Electric motor weight 22.8 kg
    # (https://www.pipistrel.fr/aircraft/electric-flight/e-811/)
    scatter_compare_mass(
        actual_mass=22.8,
        computed_mass=float(datafile["data:propulsion:he_power_train:PMSM:motor_1:mass"].value[0]),
        axes_name="E-motor mass",
        graph_number=2,
        fig=fig,
    )
    # Inverter/Power control unit and cables weight 8.1 kg
    # (https://www.pipistrel.fr/aircraft/electric-flight/e-811/)
    scatter_compare_mass(
        actual_mass=7.0,
        computed_mass=float(
            datafile["data:propulsion:he_power_train:inverter:inverter_1:mass"].value[0]
        )
        + float(
            datafile["data:propulsion:he_power_train:DC_cable_harness:harness_1:mass"].value[0]
        ),
        axes_name="Inverter mass",
        graph_number=3,
        fig=fig,
    )
    # Propeller weight 5.0 kg
    # (https://www.pipistrel.fr/aircraft/other-products/propellers/)
    scatter_compare_mass(
        actual_mass=5.0,
        computed_mass=float(
            datafile["data:propulsion:he_power_train:propeller:propeller_1:mass"].value[0]
        ),
        axes_name="Propeller mass",
        graph_number=4,
        fig=fig,
    )

    # Proper legend
    fig.add_trace(
        go.Scatter(
            mode="markers",
            y=[None],
            x=[None],
            marker=dict(color="black", symbol="circle", size=15),
            showlegend=True,
            name="Actual mass",
        )
    )
    fig.add_trace(
        go.Scatter(
            mode="markers",
            y=[None],
            x=[None],
            marker=dict(color="black", symbol="diamond", size=15),
            showlegend=True,
            name="Computed mass",
        )
    )

    # Proper title
    fig.update_layout(
        title_text="Comparison of computed mass with reference mass for the Pipistrel",
        title_x=0.5,
        xaxis_title="Component",
        yaxis_title="Mass [kg]",
        font=dict(
            size=18,
        ),
    )

    fig.show()


def scatter_compare_mass(actual_mass, computed_mass, axes_name, graph_number, fig):
    color = COLORS[graph_number % len(COLORS)]

    fig.add_trace(
        go.Scatter(
            mode="markers",
            y=[actual_mass],
            x=[axes_name],
            marker=dict(color=color, symbol="circle", size=20),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            mode="markers",
            y=[computed_mass],
            x=[axes_name],
            marker=dict(color=color, symbol="diamond", size=20),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            mode="lines",
            y=[actual_mass, computed_mass],
            x=[axes_name, axes_name],
            line=dict(color=color, width=4),
            showlegend=False,
        )
    )


def test_pipistrel_network_viewer():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "pipistrel_assembly.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "pipistrel_assembly.html")

    if not os.path.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def test_mass_bar_plot():
    results_pipistrel_file_path = pth.join(RESULTS_FOLDER_PATH, "pipistrel_out.xml")
    data_pipistrel_file_path = pth.join(DATA_FOLDER_PATH, "pipistrel_data.xml")

    fig = mass_breakdown_bar_plot(results_pipistrel_file_path, name="Computed results")
    fig = mass_breakdown_bar_plot(data_pipistrel_file_path, name="Actual value", fig=fig)

    fig.update_layout(
        title_text="Comparison of computed aircraft mass with reference value for the Pipistrel",
        title_x=0.5,
        height=800,
        width=1600,
        font_size=18,
    )
    fig.update_annotations(font_size=20)

    fig.show()


def test_aircraft_geometry_plot():
    results_pipistrel_file_path = pth.join(DATA_FOLDER_PATH, "pipistrel_for_postprocessing.xml")

    fig = aircraft_geometry_plot(results_pipistrel_file_path, name="Pipistrel")

    fig.update_layout(
        height=800,
        width=1600,
        font_size=18,
    )

    pipistrel_top_view = Image.open("data/Top_view_clean.JPG")

    fig.add_layout_image(
        dict(
            source=pipistrel_top_view,
            xref="x",
            yref="y",
            y=6.47,
            x=-10.71 / 2,
            sizex=10.71,
            sizey=6.47,
            sizing="stretch",
            opacity=0.75,
            layer="below",
        )
    )

    # fig.show()
    fig.write_image(pth.join(RESULTS_FOLDER_PATH, "pipistrel_geometry.svg"))
    fig.write_image(pth.join(RESULTS_FOLDER_PATH, "pipistrel_geometry.pdf"))


def test_plot_power_profile_results():
    mission_data_file = pth.join(RESULTS_FOLDER_PATH, "mission_data.csv")
    pt_data_file = pth.join(RESULTS_FOLDER_PATH, "pipistrel_power_train_data.csv")

    columns_to_drop = []
    for mission_variable_name in list(PROMOTION_FROM_MISSION.keys()):
        columns_to_drop.append(
            mission_variable_name + " [" + PROMOTION_FROM_MISSION[mission_variable_name] + "]"
        )

    # Read the two CSV and concatenate them so that all data can be displayed against all
    # data
    power_train_data = pd.read_csv(pt_data_file, index_col=0)
    # Remove the taxi power train data because they are not stored in the mission data
    # either
    power_train_data = power_train_data.drop([0]).iloc[:-1]
    # We readjust the index
    power_train_data = power_train_data.set_index(np.arange(len(power_train_data.index)))
    power_train_data = power_train_data.drop(columns_to_drop, axis=1)

    mission_data = pd.read_csv(mission_data_file, index_col=0)
    all_data = pd.concat([power_train_data, mission_data], axis=1)

    fig = go.Figure()

    x_name = "time"
    y_name = "propeller_1 shaft_power_in [kW]"

    name_to_flight_phase = {
        "sizing:main_route:climb": "Climb",
        "sizing:main_route:cruise": "Cruise",
        "sizing:main_route:descent": "Descent",
        "sizing:main_route:reserve": "Reserve",
    }

    for name in [
        "sizing:main_route:climb",
        "sizing:main_route:cruise",
        "sizing:main_route:descent",
        "sizing:main_route:reserve",
    ]:
        # pylint: disable=invalid-name # that's a common naming
        x = all_data.loc[all_data["name"] == name, x_name]
        # pylint: disable=invalid-name # that's a common naming
        y = all_data.loc[all_data["name"] == name, y_name]

        scatter = go.Scatter(
            x=x / 3600.0,
            y=y,
            mode="markers",
            marker={
                "color": COLOR_DICTIONARY[name],
                "symbol": MARKER_DICTIONARY[name],
                "size": 8,
            },
            name=name_to_flight_phase[name],
            legendgroup="Primary axis",
            legendgrouptitle_text="Flight phase",
        )

        fig.add_trace(scatter)

    fig.update_layout(
        xaxis_title="Flight time [h]",
        yaxis_title="Shaft power [kW]",
        showlegend=True,
    )
    fig.update_layout(
        height=800,
        width=1600,
        font_size=18,
    )
    fig.update_layout(
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        title_x=0.5,
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
    )
    fig.update_xaxes(
        ticks="outside",
        title_font=dict(size=15),
        tickfont=dict(size=15),
        showline=True,
        linecolor="black",
        linewidth=3,
        range=[-0.05, 0.9],
        dtick=0.2,
    )
    fig.update_yaxes(
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        linewidth=3,
        tickfont=dict(size=15),
        title_font=dict(size=15),
    )
    fig.update_layout(
        annotations=[
            go.layout.Annotation(
                text=" <u><b>Climb @ 650 ft/min, in ISA conditions:</b></u> <br>"
                + "POH: 48kW (Maximum Continuous Power) <br>"
                + "Computed: Around 50 kW",
                align="left",
                font=dict(size=25),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.15,
                y=0.97,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
            ),
            go.layout.Annotation(
                text=" <u><b>Cruise @ 2000 ft, 92 KIAS:</b></u> <br>"
                + "POH: 35kW <br>"
                + "Computed: 33,7kW",
                align="left",
                font=dict(size=25),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.53,
                y=0.69,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
            ),
            go.layout.Annotation(
                text=" <u><b>Reserve @ 2000 ft, 71 KIAS:</b></u> <br>"
                + "POH: 20kW <br>"
                + "Computed: 20,05kW",
                align="left",
                font=dict(size=25),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.10,
                y=0.26,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
            ),
        ],
        xaxis=dict(showgrid=True, gridcolor="lightgray", gridwidth=1),
    )
    fig.add_vline(x=0, line_width=1, line_color="lightgray")
    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()
    pdf_path = "results/pipistrel_performances_comparison.pdf"

    pio.write_image(fig, pdf_path, width=1600, height=900)
    time.sleep(3)
    pio.write_image(fig, pdf_path, width=1600, height=900)


def test_pipistrel_velis_club():
    """
    This is the thermal version of the Pipistrel Velis Electro (rather, the Pipistrel Velis Electro
    is the electric version of this aircraft.

    The design mission will be assumed to be a mission et 2000ft of altitude with 75% of
    the power which should give fuel consumption of 18.4 l/h as per Pipistrel website. The fuel
    available with the choice of 188 kg of payload leaves us 63 kg of fuel. Reserve will be
    considered as a cruise @4000ft for 30 min giving a fuel consumed of 6.32, climb fuel is computed
    based on fuel consumption at MCP for 2 min giving 0.624 kg, descent at idle fuel gives 0.31 kg
    and 1 kg of fuel for T/O and taxi is assumed, leaving 54.74 kg of fuel for cruise. The fuel
    consumption in cruise is given at 18.4 l/h in the POH or 13.248 kg/h. which means cruise will
    last for 4.05 h or 482nm. Added to that the distance for climb and descent, a design range of
    513nm is taken.

    Engine RPM:
    - Climb: 5500.
    - Cruise: 5300.
    - Descent: 2500 (Correspond to min power on the operator manual of Rotax 912).
    - Reserve: 5100. (Equivalent to a cruise at 4000ft at 65% power for 30 min) which is 116 kts

    Because what is given in the POH is most certainly the engine RPM (otherwise we would have very
    high tip mach number) and FAST-OAD-GA-HE needs propeller RPM a small conversion was made. A
    reduction ration of 2.43 was considered as per the operator manual for the rotax engine.

    I've called it the "Club" so far, but it is actually the base model we are trying to model, the
    Velis SW121, the Club is the SW121C
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_club_source.xml"
    process_file_name = "pipistrel_club_configuration.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Removing previous case and adding a recorder
    recorder_path = pth.join(RESULTS_FOLDER_PATH, "pipistrel_cases.sql")

    if pth.exists(recorder_path):
        os.remove(recorder_path)

    recorder = om.SqliteRecorder(recorder_path)
    solver = problem.model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # Run the problem
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        600.00, rel=5e-2
    )
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        349.00, rel=5e-2
    )
    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(63, rel=5e-2)


def test_pipistrel_velis_club_op():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True

    # Define used files depending on options
    process_file_name = "pipistrel_club_configuration_op.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(RESULTS_FOLDER_PATH, "pipistrel_club_out.xml")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)

    # Here we will open the input file and complete with the missing data.
    datafile = oad.DataFile(problem.input_file_path)
    datafile.append(oad.Variable("data:mission:operational:range", val=53.0, units="km"))
    datafile.append(
        oad.Variable("data:mission:operational:climb:climb_rate:sea_level", val=5.334, units="m/s")
    )
    datafile.append(
        oad.Variable(
            "data:mission:operational:climb:climb_rate:cruise_level", val=5.06476, units="m/s"
        )
    )
    datafile.append(oad.Variable("data:mission:operational:climb:v_eas", val=38.0, units="m/s"))
    datafile.append(
        oad.Variable("data:mission:operational:descent:descent_rate", val=-2.286, units="m/s")
    )
    datafile.append(oad.Variable("data:mission:operational:descent:v_eas", val=35.0, units="m/s"))
    datafile.append(
        oad.Variable("data:mission:operational:cruise:altitude", val=2000.0, units="ft")
    )
    datafile.append(oad.Variable("data:mission:operational:cruise:v_tas", val=119.0, units="knot"))
    datafile.append(
        oad.Variable("data:mission:operational:initial_climb:energy", val=0.0, units="W*h")
    )
    datafile.append(
        oad.Variable("data:mission:operational:initial_climb:fuel", val=0.0, units="kg")
    )
    datafile.append(oad.Variable("data:mission:operational:takeoff:energy", val=0.0, units="W*h"))
    datafile.append(oad.Variable("data:mission:operational:takeoff:fuel", val=0.8, units="kg"))
    datafile.append(oad.Variable("data:mission:operational:payload:CG:x", val=2.4, units="m"))
    datafile.append(oad.Variable("data:mission:operational:payload:mass", val=188.0, units="kg"))
    datafile.append(
        oad.Variable("data:mission:operational:reserve:altitude", val=1219.0, units="m")
    )
    datafile.append(
        oad.Variable("data:mission:operational:reserve:duration", val=30.0, units="min")
    )
    datafile.append(oad.Variable("data:mission:operational:taxi_in:duration", val=150.0, units="s"))
    datafile.append(oad.Variable("data:mission:operational:taxi_in:speed", val=10.28, units="m/s"))
    datafile.append(
        oad.Variable("data:mission:operational:taxi_out:duration", val=150.0, units="s")
    )
    datafile.append(oad.Variable("data:mission:operational:taxi_out:speed", val=10.28, units="m/s"))
    datafile.save()

    problem.read_inputs()
    problem.setup()

    # Run the problem
    problem.run_model()

    problem.write_outputs()


def test_plot_power_profile_results_club():
    mission_data_file = pth.join(RESULTS_FOLDER_PATH, "club_mission_data.csv")
    pt_data_file = pth.join(RESULTS_FOLDER_PATH, "pipistrel_club_pt_data.csv")

    columns_to_drop = []
    for mission_variable_name in list(PROMOTION_FROM_MISSION.keys()):
        columns_to_drop.append(
            mission_variable_name + " [" + PROMOTION_FROM_MISSION[mission_variable_name] + "]"
        )

    # Read the two CSV and concatenate them so that all data can be displayed against all
    # data
    power_train_data = pd.read_csv(pt_data_file, index_col=0)
    # Remove the taxi power train data because they are not stored in the mission data
    # either
    power_train_data = power_train_data.drop([0]).iloc[:-1]
    # We readjust the index
    power_train_data = power_train_data.set_index(np.arange(len(power_train_data.index)))
    power_train_data = power_train_data.drop(columns_to_drop, axis=1)

    mission_data = pd.read_csv(mission_data_file, index_col=0)
    all_data = pd.concat([power_train_data, mission_data], axis=1)

    fig = go.Figure()

    x_name = "time"
    y_name = "propeller_1 shaft_power_in [kW]"

    name_to_flight_phase = {
        "sizing:main_route:climb": "Climb",
        "sizing:main_route:cruise": "Cruise",
        "sizing:main_route:descent": "Descent",
        "sizing:main_route:reserve": "Reserve",
    }

    for name in [
        "sizing:main_route:climb",
        "sizing:main_route:cruise",
        "sizing:main_route:descent",
        "sizing:main_route:reserve",
    ]:
        # pylint: disable=invalid-name # that's a common naming
        x = all_data.loc[all_data["name"] == name, x_name]
        # pylint: disable=invalid-name # that's a common naming
        y = all_data.loc[all_data["name"] == name, y_name]

        scatter = go.Scatter(
            x=x / 3600.0,
            y=y,
            mode="markers",
            marker={
                "color": COLOR_DICTIONARY[name],
                "symbol": MARKER_DICTIONARY[name],
                "size": 8,
            },
            name=name_to_flight_phase[name],
            legendgroup="Primary axis",
            legendgrouptitle_text="Flight phase",
        )

        fig.add_trace(scatter)

    fig.update_layout(
        xaxis_title="Flight time [h]",
        yaxis_title="Shaft power [kW]",
        showlegend=True,
    )
    fig.update_layout(
        height=800,
        width=1600,
        font_size=18,
    )
    fig.update_layout(
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        title_x=0.5,
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
    )
    fig.update_xaxes(
        ticks="outside",
        title_font=dict(size=15),
        tickfont=dict(size=15),
        showline=True,
        linecolor="black",
        linewidth=3,
        range=[-0.1, 5.1],
        dtick=1,
    )
    fig.update_yaxes(
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        linewidth=3,
        tickfont=dict(size=15),
        title_font=dict(size=15),
    )
    fig.update_layout(
        annotations=[
            go.layout.Annotation(
                text=" <u><b>Climb @ 1020 ft/min, in ISA conditions:</b></u> <br>"
                + "POH: 69kW (Maximum Continuous Power) <br>"
                + "Computed: Around 64.5kW",
                align="left",
                font=dict(size=25),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.08,
                y=0.97,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
            ),
            go.layout.Annotation(
                text=" <u><b>Cruise @ 2000 ft, 119 KTAS:</b></u> <br>"
                + "POH: 51.8kW (75% MCP) <br>"
                + "Computed: 54kW",
                align="left",
                font=dict(size=25),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.08,
                y=0.75,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
            ),
            go.layout.Annotation(
                text=" <u><b>Reserve @ 4000 ft, 121 KTAS:</b></u> <br>"
                + "POH: 48.3kW (70% MCP)<br>"
                + "Computed: 53kW",
                align="left",
                font=dict(size=25),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.74,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
            ),
        ],
        xaxis=dict(showgrid=True, gridcolor="lightgray", gridwidth=1),
    )
    fig.add_vline(x=0, line_width=1, line_color="lightgray")
    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()
    pdf_path = "results/pipistrel_club_performances_comparison.pdf"

    pio.write_image(fig, pdf_path, width=1600, height=900)
    time.sleep(3)
    pio.write_image(fig, pdf_path, width=1600, height=900)
