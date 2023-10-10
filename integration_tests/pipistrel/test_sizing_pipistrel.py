import os
import os.path as pth
import logging

import pytest

import numpy as np

import plotly.graph_objects as go
import plotly.express as px

import fastoad.api as oad
import openmdao.api as om

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


COLORS = px.colors.qualitative.Prism


def test_pipistrel_like():

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source.xml"
    process_file_name = "pipistrel_configuration.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
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

    problem.set_val(
        "performances.solve_equilibrium.compute_dep_equilibrium.compute_energy_consumed.power_train_performances.battery_pack_1.direct_bus_connection.dc_current_out",
        units="A",
        val=np.full(92, 100.0),
    )
    problem.set_val(
        "performances.solve_equilibrium.compute_dep_equilibrium.compute_energy_consumed.power_train_performances.battery_pack_2.direct_bus_connection.dc_current_out",
        units="A",
        val=np.full(92, 100.0),
    )

    # Run the problem
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        600.00, rel=1e-2
    )


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
    # Electric motor weight 22.7 kg
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
