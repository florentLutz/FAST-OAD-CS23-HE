#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2026 ISAE-SUPAERO

import time
import pathlib

import plotly.io as pio

import fastoad.api as oad
from fastga_he.gui.lca_impact import (
    lca_impacts_bar_chart_normalised_weighted,
    lca_impacts_search_table,
    lca_impacts_sun_breakdown,
    lca_impacts_bar_chart_with_contributors,
)

RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_compare_impacts_three_designs_bar_chart_normalised():
    fig = lca_impacts_bar_chart_normalised_weighted(
        [
            RESULTS_FOLDER_PATH / "oad_process_outputs_elec_with_lca.xml",
            RESULTS_FOLDER_PATH / "oad_process_outputs_elec_lis_with_lca.xml",
            RESULTS_FOLDER_PATH / "oad_process_outputs_elec_na_ion_with_lca.xml",
        ],
        names_aircraft=[
            "2025 w/ Li-Ion",
            "2040 w/ Li-S",
            "2040 w/ Sodium-Ion",
        ],
        impact_filter_list=[
            "acidification",
            "climate change",
            "ecotoxicity freshwater",
            "energy resources non-renewable",
            "eutrophication freshwater",
            "eutrophication marine",
            "eutrophication terrestrial",
            "human toxicity carcinogenic",
            "human toxicity non-carcinogenic",
            "ionising radiation human health",
            "land use",
            "material resources metals minerals",
            "ozone depletion",
            "particulate matter formation",
            "photochemical oxidant formation human health",
            "water use",
        ],
    )
    fig.update_layout(
        title=None,
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        title_font=dict(size=20),
        legend_font=dict(size=20),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        width=1800,
        height=800,
    )
    fig.update_xaxes(
        title_font=dict(size=15),
    )
    fig.update_yaxes(title_font=dict(size=15), range=[0, 1.25e-6])
    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()

    write = True

    if write:
        pdf_path = "results/figures/impacts_evolution_elec_kodiak100.pdf"
        svg_path = "results/figures/impacts_evolution_elec_kodiak100.svg"

        pio.write_image(fig, pdf_path, width=1900, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1900, height=900)
        pio.write_image(fig, svg_path, width=1900, height=900)


def test_lca_sun_breakdown_ref():
    # Check that we can create a plot
    fig = lca_impacts_sun_breakdown(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_with_lca.xml",
        full_burst=True,
        rel="single_score",
        name_aircraft="2025 w/ Li-Ion",
    )

    fig.show()


def test_lca_sun_breakdown_li_s_and_na_ion():
    # Check that we can create a plot
    fig = lca_impacts_sun_breakdown(
        [
            RESULTS_FOLDER_PATH / "oad_process_outputs_elec_lis_with_lca.xml",
            RESULTS_FOLDER_PATH / "oad_process_outputs_elec_na_ion_with_lca.xml",
        ],
        full_burst=True,
        rel="single_score",
        name_aircraft=["2040 w/ Li-S", "2040 w/ Sodium-Ion"],
    )

    fig.show()


def test_search_engine_thesis():
    impacts_value_ref_design = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_with_lca.xml",
        ["*"],
        ["*"],
        ["battery_pack_1"],
        rel=True,
    )
    print(impacts_value_ref_design[0] * 2.0)

    impacts_value_lis_design = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_lis_with_lca.xml",
        ["*"],
        ["*"],
        ["battery_pack_1"],
        rel=True,
    )
    print(impacts_value_lis_design[0] * 2.0)

    impacts_value_na_ion_design = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_na_ion_with_lca.xml",
        ["*"],
        ["*"],
        ["battery_pack_1"],
        rel=True,
    )
    print(impacts_value_na_ion_design[0] * 2.0)

    impacts_value_na_ion_design = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_na_ion_with_lca.xml",
        ["*"],
        ["*"],
        ["dc_dc_converter_1"],
        rel=True,
    )
    print(impacts_value_na_ion_design[0] * 2.0)


def test_search_engine_airframe():
    components_list = [
        "wing",
        "fuselage",
        "horizontal_tail",
        "vertical_tail",
        "landing_gear",
        "flight_controls",
        "assembly",
    ]

    impacts_value_ref_design = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_with_lca.xml",
        ["*"] * 7,
        ["*"] * 7,
        components_list,
    )
    print(sum(impacts_value_ref_design))

    impacts_value_lis_design = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_lis_with_lca.xml",
        ["*"] * 7,
        ["*"] * 7,
        components_list,
    )
    print(sum(impacts_value_lis_design))


def test_search_engine_energy_intensity():

    design_datafile_2025 = oad.DataFile(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_with_lca.xml"
    )
    flights_per_fu_2025_design = design_datafile_2025[
        "data:environmental_impact:flight_per_fu"
    ].value[0]
    electricity_used_2025_design = design_datafile_2025[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_main_route"
    ].value[0] * 2.0
    electricity_unit = design_datafile_2025[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_main_route"
    ].units
    if electricity_unit == "W*h":
        electricity_used_2025_design /= 1000.0
    fu_per_flights_2025_design = 1.0 / flights_per_fu_2025_design
    impacts_value_2025_design = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_with_lca.xml",
        ["*"],
        ["*"],
        ["electricity_for_mission"],
        rel=False,
    )

    impact_one_flight_2025= impacts_value_2025_design[0] * fu_per_flights_2025_design
    impact_per_kwh_of_energy_used_2025 = impact_one_flight_2025 / electricity_used_2025_design
    print("2025 design, impact per kWh", impact_per_kwh_of_energy_used_2025)

    design_datafile_2040 = oad.DataFile(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_lis_with_lca.xml"
    )
    flights_per_fu_2040_design = design_datafile_2040[
        "data:environmental_impact:flight_per_fu"
    ].value[0]
    electricity_used_2040_design = design_datafile_2040[
                                       "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_main_route"
                                   ].value[0] * 2.0
    electricity_unit = design_datafile_2040[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_main_route"
    ].units
    if electricity_unit == "W*h":
        electricity_used_2040_design /= 1000.0
    fu_per_flights_2040_design = 1.0 / flights_per_fu_2040_design
    impacts_value_2040_design = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_lis_with_lca.xml",
        ["*"],
        ["*"],
        ["electricity_for_mission"],
        rel=False,
    )

    impact_one_flight_2040 = impacts_value_2040_design[0] * fu_per_flights_2040_design
    impact_per_kwh_of_energy_used_2040 = impact_one_flight_2040 / electricity_used_2040_design
    print("2040 design, impact per kWh", impact_per_kwh_of_energy_used_2040)


def test_lca_bar_chart_relative_contribution_lis():
    fig = lca_impacts_bar_chart_with_contributors(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_lis_with_lca.xml",
        name_aircraft="2040 w/ Li-S",
        impact_step="normalized",
        aggregate_and_sort_contributor={
            "Airframe": "airframe",  # Just a renaming, should work as well,
            "Electricity production": "electricity_for_mission",  # Just a renaming, should work as well,
            "Power electronics": [
                "dc_sspc_1",
                "dc_sspc_2",
                "inverter_1",
                "dc_dc_converter_1",
                "dc_dc_converter_2",
                "dc_splitter_1",
                "dc_bus_1",
            ],
            "Battery pack": ["battery_pack_1", "battery_pack_2"],
            "Others": ["propeller_1", "harness_1", "gearbox_1", "manufacturing", "distribution"],
        },
        impact_filter_list=[
            "acidification",
            "climate_change",
            "ecotoxicity_freshwater",
            "energy_resources_non-renewable",
            "eutrophication_freshwater",
            "eutrophication_marine",
            "eutrophication_terrestrial",
            "human_toxicity_carcinogenic",
            "human_toxicity_non-carcinogenic",
            "ionising_radiation_human_health",
            "land_use",
            "material_resources_metals_minerals",
            "ozone_depletion",
            "particulate_matter_formation",
            "photochemical_oxidant_formation_human_health",
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

    pdf_path = "results/figures/lis_rel_contributors.pdf"

    write = True

    if write:
        fig.update_layout(title=None)
        pio.write_image(fig, pdf_path, width=1600, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1600, height=900)


def test_lca_bar_chart_relative_contribution_na_ion():
    fig = lca_impacts_bar_chart_with_contributors(
        RESULTS_FOLDER_PATH / "oad_process_outputs_elec_na_ion_with_lca.xml",
        name_aircraft="2040 w/ Sodium-Ion",
        impact_step="normalized",
        aggregate_and_sort_contributor={
            "Airframe": "airframe",  # Just a renaming, should work as well,
            "Electricity production": "electricity_for_mission",  # Just a renaming, should work as well,
            "Power electronics": [
                "dc_sspc_1",
                "dc_sspc_2",
                "inverter_1",
                "dc_dc_converter_1",
                "dc_dc_converter_2",
                "dc_splitter_1",
                "dc_bus_1",
            ],
            "Battery pack": ["battery_pack_1", "battery_pack_2"],
            "Others": ["propeller_1", "harness_1", "gearbox_1", "manufacturing", "distribution"],
        },
        impact_filter_list=[
            "acidification",
            "climate_change",
            "ecotoxicity_freshwater",
            "energy_resources_non-renewable",
            "eutrophication_freshwater",
            "eutrophication_marine",
            "eutrophication_terrestrial",
            "human_toxicity_carcinogenic",
            "human_toxicity_non-carcinogenic",
            "ionising_radiation_human_health",
            "land_use",
            "material_resources_metals_minerals",
            "ozone_depletion",
            "particulate_matter_formation",
            "photochemical_oxidant_formation_human_health",
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

    pdf_path = "results/figures/na_ion_rel_contributors.pdf"

    write = True

    if write:
        fig.update_layout(title=None)
        pio.write_image(fig, pdf_path, width=1600, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1600, height=900)
