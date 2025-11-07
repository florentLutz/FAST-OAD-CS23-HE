#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import time
import pathlib

import logging

import numpy as np
import pytest
import plotly.graph_objects as go
import plotly.io as pio

import fastoad.api as oad

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
RESULTS_SENSITIVITY_FOLDER_PATH = pathlib.Path(__file__).parent / "results_sensitivity_plus"


def test_lca_pipistrel_reference_cell():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%, with the reference cell.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_source_with_lca.xml"
    process_file_name = "pipistrel_configuration_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # To ensure consistency with previous Pipistrel results
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:min_safe_SOC",
        units="percent",
        val=7.5,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:min_safe_SOC",
        units="percent",
        val=7.5,
    )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()


def test_lca_pipistrel_plus_reference_cell():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%, with the reference cell.
    - The front battery pack will be used only for reserve, while the back one will be used for the
        nominal mission. The LCA will be run with the battery aging model (so the two battery should
        age very differently) but the effect of SOH on performances won't be considered.
    - The LCA configuration file will be automatically generated.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_plus_source_with_lca.xml"
    process_file_name = "pipistrel_plus_configuration_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*battery_pack_1*"] = {
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
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:aging:cyclic_effect_k_factor",
        val=0.0,
    )

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # To ensure consistency with previous Pipistrel results
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:min_safe_SOC",
        units="percent",
        val=7.5,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:min_safe_SOC",
        units="percent",
        val=7.5,
    )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()


def test_lca_pipistrel_plus_reference_cell_various_soc_start():
    """
    Tests that contains the same thing as above except we will vary the starting SoC of the
    main battery to see of it affect the single score
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_plus_source_with_lca.xml"
    process_file_name = "pipistrel_plus_configuration_with_lca.yml"

    soc_start_array = np.linspace(100.0, 80.0, 11)
    for soc_start in soc_start_array:
        configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
        problem = configurator.get_problem()

        # Create inputs
        ref_inputs = DATA_FOLDER_PATH / xml_file_name

        # Setup the problem
        problem.write_needed_inputs(ref_inputs)
        problem.read_inputs()

        problem.model_options["*battery_pack_1*"] = {
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
            "data:propulsion:he_power_train:battery_pack:battery_pack_2:aging:cyclic_effect_k_factor",
            val=0.0,
        )

        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start",
            units="percent",
            val=soc_start,
        )

        # Give good initial guess on a few key value to reduce the time it takes to converge
        problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
        problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
        problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
        problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
        problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

        # To ensure consistency with previous Pipistrel results
        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:min_safe_SOC",
            units="percent",
            val=7.5,
        )
        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_2:min_safe_SOC",
            units="percent",
            val=7.5,
        )

        # Run the problem
        problem.run_model()

        # Write the outputs
        file_name = str(int(soc_start)) + "_soc_start_out.xml"
        problem.output_file_path = RESULTS_SENSITIVITY_FOLDER_PATH / file_name
        problem.write_outputs()


def test_post_process_results_pipistrel_plus():
    fig = go.Figure()

    socs_start_mission = []
    single_scores = []
    battery_mass_per_fu = []
    battery_mass = []

    generic_list = []

    for file in RESULTS_SENSITIVITY_FOLDER_PATH.iterdir():
        if file.is_file():
            datafile = oad.DataFile(RESULTS_SENSITIVITY_FOLDER_PATH / file)
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
                datafile[
                    "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan"
                ].value[0]
            )

    ref_case_datafile = oad.DataFile(RESULTS_FOLDER_PATH / "pipistrel_out_with_lca.xml")

    sorted_single_scores = [single_scores[i] for i in np.argsort(socs_start_mission)]
    sorted_battery_mass_per_fu = [battery_mass_per_fu[i] for i in np.argsort(socs_start_mission)]
    sorted_socs_start = np.sort(socs_start_mission)

    sorted_generic_list = [generic_list[i] for i in np.argsort(socs_start_mission)]
    # sorted_battery_mass = [battery_mass[i] for i in np.argsort(socs_start_mission)]

    ref_case_single_score_adim = (
        ref_case_datafile["data:environmental_impact:single_score"].value[0]
        / sorted_single_scores[-1]
    )

    scatter = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_single_scores) / sorted_single_scores[-1],
        mode="lines+markers",
        name="Single score",
        marker=dict(color="black", size=15, symbol="circle"),
        line=dict(color="black", width=3),
    )
    fig.add_trace(scatter)
    scatter_2 = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_battery_mass_per_fu) / sorted_battery_mass_per_fu[-1],
        mode="lines+markers",
        name="Battery mass per functional unit",
        marker=dict(color="red", size=15, symbol="diamond"),
        line=dict(color="red", width=3),
    )
    fig.add_trace(scatter_2)
    scatter_3 = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_generic_list) / sorted_generic_list[-1],
        mode="lines+markers",
        name="Battery lifespan",
        marker=dict(color="grey", size=15, symbol="cross"),
        line=dict(color="grey", width=3),
    )
    fig.add_trace(scatter_3)

    fig.update_layout(
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        height=800,
        width=1600,
        margin=dict(l=5, r=5, t=60, b=5),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title_font=dict(size=20),
        tickfont=dict(size=20),
        title="Initial SoC on the design mission [%]",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title_font=dict(size=20),
        tickfont=dict(size=20),
        title="Variation with respect to 100% SoC case",
    )
    fig.add_hline(
        y=ref_case_single_score_adim,
        line_width=1,
        line_dash="dash",
        line_color="red",
        annotation_text="Environmental single score of the original design",
        annotation_position="bottom right",
        annotation={"font": {"size": 17, "color": "red"}, "align": "right"},
    )

    fig.add_annotation(
        x=100.0,
        y=1.05,
        text="16 batteries required during the<br>aircraft lifespan",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=17, color="black"),
        align="right",
    )
    _draw_arrow(
        figure=fig,
        arrow_width=3,
        arrow_start_coords=(99.0,1.05),
        arrow_end_coords=(99.9,1.005),
        arrow_color="black",
    )

    fig.add_annotation(
        x=98.0,
        y=0.9614287-0.008,
        text="15 batteries required during the<br>aircraft lifespan",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=17, color="black"),
        align="center",
    )

    fig.add_annotation(
        x=92.0,
        y=0.9708026-0.008,
        text="14 batteries required during the<br>aircraft lifespan",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=17, color="black"),
        align="center",
    )

    fig.add_annotation(
        x=86.0,
        y=0.981339 - 0.008,
        text="13 batteries required during the<br>aircraft lifespan",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=17, color="black"),
        align="center",
    )

    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()

    write = True

    if write:
        fig.update_layout(title=None)
        pdf_path = "results/pipistrel_plus_soc_start_variation.pdf"

        pio.write_image(fig, pdf_path, width=1900, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1900, height=900)


def test_lca_pipistrel_plus_plus_high_bed_cell():
    """
    Tests that contains:
    - A full sizing of a Pipistrel Velis Electro with an enforce of the cell number (not exactly the
        reference case), but with a starting SOC of 100%, with the reference cell.
    - The front battery pack will be used only for reserve, while the back one will be used for the
        nominal mission. The LCA will be run with the battery aging model (so the two battery should
        age very differently) but the effect of SOH on performances won't be considered.
    - The front battery will use a high energy density cell
    - The LCA configuration file won't be automatically generated.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_plus_source_with_lca.xml"
    process_file_name = "pipistrel_plus_plus_configuration_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*battery_pack_1*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.model_options["*battery_pack_2*"] = {
        "cell_capacity_ref": 1.34,
        "cell_weight_ref": 11.7e-3,
        "reference_curve_current": [100.0, 1000.0, 3000.0, 5100.0],
        "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
    }

    problem.setup()

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
        val=4.0,
        units="h**-1",
    )

    # TODO: This case will work because it is reasonable to assume that the high BED cell has the
    # TODO: same polarization curve as the reference cell otherwise we would have needed to change
    # TODO: the submodels for the polarization curve but since they are shared it would have caused
    # TODO: a problem.

    # Also we assume same aging mechanism
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:aging:cyclic_effect_k_factor",
        val=0.0,
    )

    # Give good initial guess on a few key value to reduce the time it takes to converge
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=400.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:ZFW", units="kg", val=600.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=600.0)

    # To ensure consistency with previous Pipistrel results
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:min_safe_SOC",
        units="percent",
        val=7.5,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:min_safe_SOC",
        units="percent",
        val=7.5,
    )

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()


def test_only_lca_pipistrel_reference_cell_pessimistic():
    """
    Tests that contains:
    - An LCA evaluation of the reference Pipistrel with a buy to fly ratio of 2 for composite and
        a European electric mix
    - OAD results from the first test will be used.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_out_with_lca.xml"
    process_file_name = "pipistrel_configuration_only_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val("data:environmental_impact:buy_to_fly:composite", val=2.0)

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.0036902, rel=1e-3
    )


def test_only_lca_pipistrel_plus_plus_pessimistic():
    """
    Tests that contains:
    - An LCA evaluation of the Pipistrel plus plus with a buy to fly ratio of 2 for composite and
        a European electric mix
    - OAD results from the test_lca_pipistrel_plus_plus_high_bed_cell will be used.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "pipistrel_plus_plus_out_with_lca.xml"
    process_file_name = "pipistrel_plus_plus_configuration_only_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = RESULTS_FOLDER_PATH / xml_file_name

    # Setup the problem
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.set_val("data:environmental_impact:buy_to_fly:composite", val=2.0)

    # Run the problem
    problem.run_model()

    # Write the outputs
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.00352396, rel=1e-3
    )


def _draw_arrow(figure, arrow_width, arrow_start_coords, arrow_end_coords, arrow_color):
    loc_arrowhead_x, loc_arrowhead_y = arrow_end_coords
    loc_arrowtail_x, loc_arrowtail_y = arrow_start_coords
    loc_segment_percentage = 0.95

    figure.add_annotation(
        x=loc_arrowhead_x,
        y=loc_arrowhead_y,
        ax=loc_arrowtail_x,
        ay=loc_arrowtail_y,
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=arrow_width,
        startarrowsize=arrow_width,
        arrowcolor=arrow_color,
    )
    figure.add_shape(
        type="line",
        x0=loc_arrowtail_x - loc_segment_percentage * (loc_arrowtail_x - loc_arrowhead_x),
        x1=loc_arrowtail_x,
        y0=loc_arrowtail_y + loc_segment_percentage * (loc_arrowhead_y - loc_arrowtail_y),
        y1=loc_arrowtail_y,
        xref="x",
        yref="y",
        line=dict(width=arrow_width, color=arrow_color),
    )
