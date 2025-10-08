#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import os
import pathlib
import time
import pytest

import plotly.io as pio

from fastga_he.gui.lca_impact import lca_impacts_bar_chart_with_contributors, lca_impacts_bar_chart_simple

RESULT_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
PICTURE_FOLDER_PATH = pathlib.Path(__file__).parent / "results_figures"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_relative_contribution_ref_cirrus_sr22():
    fig = lca_impacts_bar_chart_with_contributors(
        RESULT_FOLDER_PATH / "full_sizing_out_with_lca.xml",
        name_aircraft="the reference Cirrus SR22",
        detailed_component_contributions=True,
        legend_rename={
            "manufacturing": "Line testing",
            "distribution": "Distribution",
        },
        aggregate_and_sort_contributor={
            "Powertrain production": [
                "propeller_1: production",
                "fuel_system_1: production",
                "ice_1: production",
            ],
            "AvGas combustion": ["ice_1: operation"],
            "AvGas production": ["gasoline_for_mission: operation"],
            "Airframe production": ["airframe: production"]
        },
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
            "total_ecosystem_quality",
            "total_human_health",
            "total_natural_resources",
        ],
    )

    fig.update_layout(height=800, width=1600, title=None, margin=dict(l=5, r=5, t=60, b=5),)
    pio.write_image(fig, PICTURE_FOLDER_PATH / "ref_cirrus_sr22_relative_contribution.pdf",  width=1600, height=900)
    time.sleep(3)
    pio.write_image(fig, PICTURE_FOLDER_PATH / "ref_cirrus_sr22_relative_contribution.pdf",  width=1600, height=900)

    fig.show()

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_relative_contribution_cirrus_esr22():
    fig = lca_impacts_bar_chart_with_contributors(
        RESULT_FOLDER_PATH / "full_sizing_elec_out_two_motors_with_lca.xml",
        name_aircraft="the electric SR22",
        detailed_component_contributions=True,
        legend_rename={
            "manufacturing": "Line testing",
            "distribution": "Distribution",
        },
        # aggregate_phase=["production"],
        aggregate_and_sort_contributor={
            "Rest of powertrain production": [
                "propeller_1: production",
                "planetary_gear_1: production",
                "motor_1: production",
                "motor_2: production",
                "inverter_1: production",
                "inverter_2: production",
                "harness_1: production",
                "dc_sspc_1: production",
                "dc_sspc_2: production",
                "dc_splitter_1: production",
                "dc_bus_1: production",
            ],
            "Electricity production": ["electricity_for_mission: operation"],
            "Airframe production": ["airframe: production"],
            "Battery production": ["battery_pack_1: production", "battery_pack_2: production"]
        },
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
            "total_ecosystem_quality",
            "total_human_health",
            "total_natural_resources",
        ],
    )

    fig.update_layout(height=800, width=1600, title=None, margin=dict(l=5, r=5, t=60, b=5))
    pio.write_image(fig, PICTURE_FOLDER_PATH / "cirrus_esr22_relative_contribution.pdf",  width=1600, height=900)
    time.sleep(3)
    pio.write_image(fig, PICTURE_FOLDER_PATH / "cirrus_esr22_relative_contribution.pdf",  width=1600, height=900)

    fig.show()


def test_compare_normalized_endpoints():

    fig = lca_impacts_bar_chart_simple(
        [
            RESULT_FOLDER_PATH / "full_sizing_out_with_lca.xml",
            RESULT_FOLDER_PATH / "full_sizing_elec_out_two_motors_with_lca.xml",
        ],
        names_aircraft=[
            "Reference Cirrus SR22",
            "Electric eSR22",
        ],
    )

    fig.show()