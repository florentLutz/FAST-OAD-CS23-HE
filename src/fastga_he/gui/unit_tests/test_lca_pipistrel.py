#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import time
import os
import pathlib

import pytest

import plotly.io as pio

from ..lca_impact import (
    lca_impacts_bar_chart_simple,
    lca_impacts_sun_breakdown,
    lca_impacts_bar_chart_normalised,
    lca_impacts_bar_chart_with_contributors,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca_pipistrel"
FIGURE_FOLDER_PATH = pathlib.Path(__file__).parent / "results"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart():
    fig = lca_impacts_bar_chart_simple(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_eu_mix.xml",
        ],
        names_aircraft=[
            "Pipistrel SW121",
            "Pipistrel Velis Electro (FR mix)",
            "Pipistrel Velis Electro (EU mix)",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_sun_breakdown_pipistrel_presentation():
    # Check that we can create a plot
    fig = lca_impacts_sun_breakdown(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out.xml",
        ],
        full_burst=True,
        rel="single_score",
        name_aircraft=["Thermal propulsion", "Electric Propulsion"],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized():
    fig = lca_impacts_bar_chart_normalised(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out_recipe.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_recipe.xml",
        ],
        names_aircraft=[
            "Pipistrel SW121",
            "Pipistrel Velis Electro",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized_ef():
    fig = lca_impacts_bar_chart_normalised(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out.xml",
        ],
        names_aircraft=[
            "Pipistrel SW121",
            "Pipistrel Velis Electro",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized_arvidsson():
    fig = lca_impacts_bar_chart_normalised(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out_arvidsson.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_arvidsson.xml",
        ],
        names_aircraft=[
            "Pipistrel Alpha Trainer",
            "Pipistrel Alpha Electro",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_absolute_phase_hybrid():
    fig = lca_impacts_bar_chart_with_contributors(
        DATA_FOLDER_PATH / "pipistrel_electro_lca_out_recipe.xml",
        name_aircraft="pipistrel Velis Electro",
        impact_step="normalized",
        impact_filter_list=[
            "ecotoxicity_freshwater",
            "ecotoxicity_marine",
            "human_toxicity_carcinogenic",
            "energy_resources_non-renewablefossil",
            "climate_change",
        ],
        aggregate_and_sort_contributor={
            "Airframe": "airframe",
            "Battery pack": ["battery_pack_1", "battery_pack_2"],
            "Others": [
                "motor_1",
                "inverter_1",
                "harness_1",
                "dc_sspc_1",
                "dc_sspc_2",
                "dc_splitter_1",
                "dc_bus_1",
                "manufacturing",
                "distribution",
            ],
            "Use phase": "electricity_for_mission",
            "Propeller": "propeller_1",
        },
    )
    fig.update_layout(title_text=None, height=800, width=1000)
    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized_comparison_with_heavy():
    fig = lca_impacts_bar_chart_simple(
        [
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_recipe_upd.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_recipe.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_recipe_btf.xml",
        ],
        names_aircraft=[
            "Pipistrel Velis Electro (composite version, buy-to-fly=1)",
            "Pipistrel Velis Electro (metallic version, buy-to-fly=1)",
            "Pipistrel Velis Electro (metallic version, buy-to-fly=7.5)",
        ],
        impact_step="normalized",
        graph_title="Comparison of Pipistrel version with different materials for the airframe",
    )

    fig.show()


def test_lca_bar_chart_absolute_phase_pipistrel_heavy():
    fig = lca_impacts_bar_chart_with_contributors(
        DATA_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_recipe_btf.xml",
        name_aircraft="Pipistrel Velis Electro (metallic version, buy-to-fly=7.5)",
        impact_step="normalized",
        impact_filter_list=[
            "photochemical_oxidant_formation_terrestrial_ecosystems",
            "human_toxicity_carcinogenic",
            "energy_resources_non-renewablefossil",
            "climate_change",
        ],
        aggregate_and_sort_contributor={
            "Airframe": "airframe",
            "Battery pack": ["battery_pack_1", "battery_pack_2"],
            "Others": [
                "motor_1",
                "inverter_1",
                "harness_1",
                "dc_sspc_1",
                "dc_sspc_2",
                "dc_splitter_1",
                "dc_bus_1",
                "manufacturing",
                "distribution",
            ],
            "Use phase": "electricity_for_mission",
            "Propeller": "propeller_1",
        },
    )
    fig.update_layout(title_text=None, height=800, width=1000)
    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized_comparison_with_heavy_btf_both():
    fig = lca_impacts_bar_chart_simple(
        [
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_recipe_fr_mix_btf.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_recipe_fr_mix_btf.xml",
        ],
        names_aircraft=[
            "Pipistrel Velis Electro<br>(composite version, buy-to-fly=1.5)",
            "Pipistrel Velis Electro<br>(metallic version, buy-to-fly=7.5)",
        ],
        impact_step="normalized",
        graph_title="Comparison of Pipistrel version with different materials for the airframe",
        impact_filter_list=[
            "acidification terrestrial",
            "climate change",
            "ecotoxicity freshwater",
            "ecotoxicity marine",
            "ecotoxicity terrestrial",
            "energy resources non-renewablefossil",
            "eutrophication freshwater",
            "eutrophication marine",
            "human toxicity carcinogenic",
            "human toxicity non-carcinogenic",
            "ionising radiation",
            "land use",
            "material resources metals minerals",
            "ozone depletion",
            "particulate matter formation",
            "photochemical oxidant formation human health",
            "photochemical oxidant formation terrestrial ecosystems",
            "water use",
        ],
    )
    fig.update_layout(
        title=None,
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        title_font=dict(size=20),
        legend_font=dict(size=20),
        # legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.0),
        width=1800,
        height=800,
    )
    fig.update_xaxes(
        title_font=dict(size=15),
    )
    fig.update_yaxes(
        title_font=dict(size=15),
    )
    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()

    pdf_path = FIGURE_FOLDER_PATH / "pipistrel_velis_vs_heavy.pdf"

    write = True

    if write:
        fig.update_layout(title=None)
        pio.write_image(fig, pdf_path, width=1600, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1600, height=900)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized_comparison_with_heavy_btf_both_eu_mix():
    fig = lca_impacts_bar_chart_simple(
        [
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_eu_mix_btf.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_eu_mix_btf.xml",
        ],
        names_aircraft=[
            "Pipistrel Velis Electro (composite version, buy-to-fly=1.5)",
            "Pipistrel Velis Electro (metallic version, buy-to-fly=7.5)",
        ],
        impact_step="normalized",
        graph_title="Comparison of Pipistrel version with different materials for the airframe, with an EU mix",
    )

    fig.show()


def test_lca_bar_chart_relative_contribution_reference():
    fig = lca_impacts_bar_chart_with_contributors(
        DATA_FOLDER_PATH / "pipistrel_electro_lca_out_recipe_fr_mix_btf.xml",
        name_aircraft="Pipistrel Velis Electro<br>(composite version, buy-to-fly=1.5)",
        impact_step="normalized",
        impact_filter_list=[
            "acidification_terrestrial",
            "climate_change",
            "ecotoxicity_freshwater",
            "ecotoxicity_marine",
            "ecotoxicity_terrestrial",
            "energy_resources_non-renewablefossil",
            "eutrophication_freshwater",
            "eutrophication_marine",
            "human_toxicity_carcinogenic",
            "human_toxicity_non-carcinogenic",
            "ionising_radiation",
            "land_use",
            "material_resources_metals_minerals",
            "ozone_depletion",
            "particulate_matter_formation",
            "photochemical_oxidant_formation_human_health",
            "photochemical_oxidant_formation_terrestrial_ecosystems",
            "water_use",
        ],
    )

    fig.update_layout(
        title=None,
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        title_font=dict(size=20),
        legend_font=dict(size=20),
        # legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.0),
        width=1800,
        height=800,
    )
    fig.update_xaxes(
        title_font=dict(size=15),
    )
    fig.update_yaxes(
        title_font=dict(size=15),
    )
    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()
