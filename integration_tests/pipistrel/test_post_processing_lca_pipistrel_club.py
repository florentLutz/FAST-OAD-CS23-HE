#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import time
import pathlib

import plotly.io as pio


from fastga_he.gui.lca_impact import (
    lca_impacts_bar_chart_simple,
    lca_raw_impact_comparison_advanced,
    lca_impacts_bar_chart_with_contributors,
)

RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_compare_impacts_designs_simple_bar_chart():
    fig = lca_impacts_bar_chart_simple(
        [
            RESULTS_FOLDER_PATH / "pipistrel_club_with_lca_out.xml",
            RESULTS_FOLDER_PATH / "pipistrel_club_heavy_with_lca_out.xml",
        ],
        names_aircraft=[
            "Pipistrel Club<br>(composite version, buy-to-fly=1.5)",
            "Pipistrel Club<br>(metallic version, buy-to-fly=7.5)",
        ],
        impact_step="normalized",
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

    pdf_path = "results/pipistrel_sw121_vs_heavy.pdf"

    write = False

    if write:
        fig.update_layout(title=None)
        pio.write_image(fig, pdf_path, width=1600, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1600, height=900)


def test_compare_impacts_designs_simple_bar_chart_only_endpoints():
    fig = lca_impacts_bar_chart_simple(
        [
            RESULTS_FOLDER_PATH / "pipistrel_club_with_lca_out.xml",
            RESULTS_FOLDER_PATH / "pipistrel_club_heavy_with_lca_out.xml",
        ],
        names_aircraft=[
            "Pipistrel Club<br>(composite version, buy-to-fly=1.5)",
            "Pipistrel Club<br>(metallic version, buy-to-fly=7.5)",
        ],
        impact_step="normalized",
        impact_filter_list=[
            "total ecosystem quality",
            "total human health",
            "total natural resources",
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


def test_compare_impacts_designs_with_contributor():
    fig = lca_raw_impact_comparison_advanced(
        [
            RESULTS_FOLDER_PATH / "pipistrel_club_with_lca_out.xml",
            RESULTS_FOLDER_PATH / "pipistrel_club_heavy_with_lca_out.xml",
        ],
        names_aircraft=[
            "Reference Pipistrel Club",
            "Heavy Pipistrel Club",
        ],
        impact_category="human toxicity non-carcinogenic",
        aggregate_and_sort_contributor={
            "Airframe": "airframe",  # Just a renaming, should work as well,
            "In flight emissions": "ice_1",  # Just a renaming, should work as well,
            "Fuel production": "gasoline_for_mission",  # Just a renaming, should work as well,
            "Others": [
                "propeller_1",
                "fuel_system_1",
                "manufacturing",
                "distribution",
            ],
        },
    )
    fig.update_layout(width=1000)

    fig.show()


def test_lca_bar_chart_relative_contribution_original_design():
    fig = lca_impacts_bar_chart_with_contributors(
        RESULTS_FOLDER_PATH / "pipistrel_club_with_lca_out.xml",
        name_aircraft="Reference Pipistrel Club",
        impact_step="normalized",
        aggregate_and_sort_contributor={
            "Airframe": "airframe",  # Just a renaming, should work as well,
            "Fuel combustion": "ice_1",  # Just a renaming, should work as well,
            "Fuel production": "gasoline_for_mission",  # Just a renaming, should work as well,
            "Others": [
                "propeller_1",
                "fuel_system_1",
                "manufacturing",
                "distribution",
            ],
        },
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

    pdf_path = "results/pipistrel_sw121_rel_contributors.pdf"

    write = True

    if write:
        fig.update_layout(title=None)
        pio.write_image(fig, pdf_path, width=1600, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1600, height=900)


def test_lca_bar_chart_relative_contribution_heavy():
    fig = lca_impacts_bar_chart_with_contributors(
        RESULTS_FOLDER_PATH / "pipistrel_club_heavy_with_lca_out.xml",
        name_aircraft="Pipistrel Club Heavy",
        impact_step="normalized",
        aggregate_and_sort_contributor={
            "Airframe": "airframe",  # Just a renaming, should work as well,
            "Fuel combustion": "ice_1",  # Just a renaming, should work as well,
            "Fuel production": "gasoline_for_mission",  # Just a renaming, should work as well,
            "Others": [
                "propeller_1",
                "fuel_system_1",
                "manufacturing",
                "distribution",
            ],
        },
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
