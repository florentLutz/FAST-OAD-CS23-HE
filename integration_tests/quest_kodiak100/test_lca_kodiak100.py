#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import time
import os
import pathlib
import logging
import pytest

import numpy as np
import fastoad.api as oad

import plotly.graph_objects as go

from fastga_he.gui.lca_impact import (
    lca_score_sensitivity_simple,
    lca_impacts_bar_chart_simple,
    lca_impacts_bar_chart_with_components_absolute,
    lca_impacts_search_table,
)

DATA_WO_LCA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
WORKDIR_FOLDER_PATH = pathlib.Path(__file__).parent / "workdir"
FIGURE_FOLDER_PATH = pathlib.Path(__file__).parent / "results" / "figures"
RESULTS_SENSITIVITY_FOLDER_PATH = pathlib.Path(__file__).parent / "results" / "lifespan_sensitivity"
RESULTS_SENSITIVITY_FOLDER_PATH_REF = (
    pathlib.Path(__file__).parent / "results" / "lifespan_sensitivity_ref"
)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_lifespan_sensitivity_reference_kodiak100():
    """
    Run the LCA on the result sizing for the hybrid Kodiak and varies the expected lifespan of the
    design
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Please note that this input file is the output file of the sizing process
    xml_file_name = "ref_kodiak_lca.xml"
    process_file_name = "ref_kodiak_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # As we are not doing any sizing, the change in model option we did in the sizing process is not
    # mandatory, so it is not required
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    mean_airframe_hours = 3524.9
    std_airframe_hours = 266.5

    problem.run_model()

    for airframe_hours in np.linspace(
        mean_airframe_hours - 3.0 * std_airframe_hours,
        2.0 * mean_airframe_hours + 6.0 * std_airframe_hours,
    ):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        problem.run_model()

        file_name = xml_file_name + "_" + str(int(airframe_hours)) + ".xml"
        problem.output_file_path = RESULTS_SENSITIVITY_FOLDER_PATH_REF / file_name
        problem.write_outputs()


def test_lifespan_sensitivity_hybrid_kodiak100():
    """
    Run the LCA on the result sizing for the hybrid Kodiak and varies the expected lifespan of the
    design
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Please note that this input file is the output file of the sizing process
    xml_file_name = "hybrid_kodiak_lca.xml"
    process_file_name = "hybrid_kodiak_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_FOLDER_PATH / xml_file_name

    # As we are not doing any sizing, the change in model option we did in the sizing process is not
    # mandatory, so it is not required
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.setup()

    mean_airframe_hours = 3524.9
    std_airframe_hours = 266.5

    problem.run_model()

    for airframe_hours in np.linspace(
        mean_airframe_hours - 3.0 * std_airframe_hours,
        2.0 * mean_airframe_hours + 6.0 * std_airframe_hours,
    ):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        problem.run_model()

        file_name = xml_file_name + "_" + str(int(airframe_hours)) + ".xml"
        problem.output_file_path = RESULTS_SENSITIVITY_FOLDER_PATH / file_name
        problem.write_outputs()


def test_hybrid_kodiak_sizing_with_lca_current_tech():
    """
    This is the case I'm presenting in my thesis, I'm just re-adding it for convenience since the
    future scenario will be below.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_kodiak_with_lca_current.xml"
    process_file_name = "hybrid_kodiak_retrofit_with_lca_current.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_WO_LCA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_WO_LCA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # om.n2(problem, outfile=pth.join(RESULTS_FOLDER_PATH, "hybrid_kodiak_n2.html"))

    # Change battery pack characteristics so that they match those of a high power,
    # lower capacity cell like the Samsung INR18650-25R, we also take the weight fraction of the
    # Pipistrel battery. Assumes same polarization curve
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

    # I want to correct some input data without changing the input file which might be shared
    problem.set_val("data:TLAR:max_airframe_hours", val=7077.0, units="h")
    problem.set_val("data:environmental_impact:buy_to_fly:metallic", val=7.5)

    problem.run_model()
    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "hybrid_kodiak_sizing_with_lca_current_tech_out.xml"
    )
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        7.981102867555266e-06, rel=1e-2
    )


def test_hybrid_kodiak_sizing_with_lca_future_tech():
    """
    This test does the sizing of a future hybrid aircraft with similar requirement to the Kodiak
    100. A design from scratch approach will be adopted as we don't expect rapid EIS. Given the
    results on the existing Kodiak 100 and seeing the improvement possible, a feasible
    design in highly plausible.  Max payload was selected with a range of 600 nm covering 99% of
    the mission of the Kodiak 100. Please note this is outside the capabilities of the Kodiak
    100 which is limited to 400 nm at this payload. Future tech will assume:
    - SiC based power converters with higher switching frequencies.
    - Future materials for electric motors
    - Na-Ion batteries with practical energy densities of 700 Wh/kg and lifecycles of 12000 cycles
    - For lack of accurate process for the Na-Ion LCA the default process will be used. It is worth
    noting that doing so we will over-estimate the score of the battery pack as Sodium Ion batteries
    are expected to have lower impacts (and costs)
    - The airframe will be assumed to be made with additive manufacturing thus a BtF of 1 for
    metallic materials
    - An improvement of the turboshaft's sfc of -15% will be assumed according to the numbers
    presented in Clean Sky's 2 Next gen small Turboprop (https://clean-aviation.eu/research-and-
    innovation/clean-sky-2/clean-sky-2-achievement-report/clean-sky-2s-key-achievements/next-gen-
    small-turboprop)
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_kodiak_with_lca_future.xml"
    process_file_name = "hybrid_kodiak_full_sizing_with_lca_future.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_WO_LCA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_WO_LCA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # om.n2(problem, outfile=pth.join(RESULTS_FOLDER_PATH, "hybrid_kodiak_n2.html"))

    # Change battery pack characteristics so that they match those of a high power,
    # lower capacity cell like the Samsung INR18650-25R, we also take the weight fraction of the
    # Pipistrel battery. Adjust the cell weight so that it reaches energy densities close to values
    # matching a Sodium-Ion cell.
    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 10.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.setup()

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=30.0
    )
    problem.set_val("data:weight:aircraft:MTOW", val=3000.0, units="kg")
    problem.set_val("data:geometry:wing:area", val=22.5, units="m**2")

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )

    # I want to correct some input data without changing the input file which might be shared
    problem.set_val("data:TLAR:max_airframe_hours", val=7077.0, units="h")
    problem.set_val("data:environmental_impact:buy_to_fly:metallic", val=1.0)
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan", val=12000.0
    )

    problem.run_model()
    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "hybrid_kodiak_sizing_with_lca_future_tech_out.xml"
    )
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        5.79842067e-06, rel=1e-2
    )


def test_thermal_kodiak_sizing_with_lca_future_tech():
    """
    This test does the sizing of a future thermal aircraft with similar requirement to the Kodiak
    100. Max payload was selected with a range of 600 nm covering 99% of the mission of the Kodiak
    100. Please note this is outside the capabilities of the Kodiak 100 which is limited to
    400 nm at this payload. Future tech will assume:
    - The airframe will be assumed to be made with additive manufacturing thus a BtF of 1 for
    metallic materials
    - An improvement of the turboshaft's sfc of -15% will be assumed according to the numbers
    presented in Clean Sky's 2 Next gen small Turboprop (https://clean-aviation.eu/research-and-
    innovation/clean-sky-2/clean-sky-2-achievement-report/clean-sky-2s-key-achievements/next-gen-
    small-turboprop)
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "input_kodiak100_with_lca_future.xml"
    process_file_name = "full_sizing_kodiak100_with_lca_future.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_WO_LCA_FOLDER_PATH / process_file_name)
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = DATA_WO_LCA_FOLDER_PATH / xml_file_name

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # I want to correct some input data without changing the input file which might be shared
    problem.set_val("data:TLAR:max_airframe_hours", val=7077.0, units="h")
    problem.set_val("data:environmental_impact:buy_to_fly:metallic", val=1.0)

    problem.run_model()
    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "thermal_kodiak_sizing_with_lca_future_tech_out.xml"
    )
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        7.86589636e-06, rel=1e-2
    )
    # For the aircraft with current tech on same design mission it 9.52e-06 and not 8 something as
    # on the other case due to the projections made by premise.


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_single_score_sensitivity_analysis_two_plots():
    # Check that we can create a plot
    fig = lca_score_sensitivity_simple(
        results_folder_path=RESULTS_SENSITIVITY_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
    )

    fig = lca_score_sensitivity_simple(
        results_folder_path=RESULTS_SENSITIVITY_FOLDER_PATH_REF,
        prefix="ref_kodiak",
        name="Reference Kodiak",
        fig=fig,
    )

    # We do that so that the legend doesn't overlap the y-axis, which as a reminder, we have to
    # put on the right otherwise we can't change the font without changing the yaxis range
    fig.update_xaxes(domain=[0, 0.95])

    fig = go.FigureWidget(fig)

    fig.show()

    fig.update_layout(height=800.0, width=1600.0, title=None)
    fig.write_image(FIGURE_FOLDER_PATH / "ga_single_score_evolution.pdf")
    time.sleep(3)
    fig.write_image(FIGURE_FOLDER_PATH / "ga_single_score_evolution.pdf")
    fig.write_image(FIGURE_FOLDER_PATH / "ga_single_score_evolution.svg")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_relative_paper():
    fig = lca_impacts_bar_chart_simple(
        [
            RESULTS_SENSITIVITY_FOLDER_PATH_REF / "ref_kodiak_lca.xml_7077.xml",
            RESULTS_SENSITIVITY_FOLDER_PATH / "hybrid_kodiak_lca.xml_7077.xml",
        ],
        names_aircraft=["Reference Kodiak 100", "Hybrid Kodiak 100"],
    )
    fig.update_layout(title_text=None)

    fig.show()
    fig.update_layout(height=800, width=1600)
    fig.write_image(FIGURE_FOLDER_PATH / "ga_impacts_evolution.pdf")
    time.sleep(3)
    fig.write_image(FIGURE_FOLDER_PATH / "ga_impacts_evolution.pdf")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_absolute_phase():
    fig = lca_impacts_bar_chart_with_components_absolute(
        RESULTS_SENSITIVITY_FOLDER_PATH_REF / "ref_kodiak_lca.xml_7077.xml",
        name_aircraft="Reference Kodiak 100",
        detailed_component_contributions=True,
        legend_rename={
            "manufacturing": "line testing",
            "turboshaft: operation": "kerosene combustion",
            "kerosene for mission: operation": "kerosene production",
        },
        aggregate_phase=["production"],
    )
    fig.update_layout(title_text=None)

    fig.show()
    fig.update_layout(height=800, width=1600)
    # Somehow this prevents the ugly footer from appearing !
    fig.write_image(FIGURE_FOLDER_PATH / "ga_component_contribution_ref.pdf")
    time.sleep(3)
    fig.write_image(FIGURE_FOLDER_PATH / "ga_component_contribution_ref.pdf")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_search_engine_thesis():
    # For the reference design, we consider the impacts of the fuel consumption and the production
    # of the fuel. Last one is sanity check.
    # Results are naturally given for a payload*km, but since we want the impact per MJ of primary
    # energy used, we must first put them back to the frame of one flight and then transfer to one MJ of that flight

    ref_design_datafile = oad.DataFile(
        RESULTS_SENSITIVITY_FOLDER_PATH_REF / "ref_kodiak_lca.xml_7077.xml"
    )
    flights_per_fu_ref_design = ref_design_datafile[
        "data:environmental_impact:flight_per_fu"
    ].value[0]
    print("Flights per FU ref design", flights_per_fu_ref_design)
    print("FU per flights ref design", 1.0 / flights_per_fu_ref_design)

    single_score_ref = ref_design_datafile["data:environmental_impact:single_score"].value[0]

    # Not available here directly, have to rely on the amount of kerosene in the tanks
    fuel_burned_ref_design = (
        ref_design_datafile[
            "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_main_route"
        ].value[0]
        * 2.0
    )
    fuel_burned_per_pax_km_ref_design = fuel_burned_ref_design * flights_per_fu_ref_design
    energy_mission_ref_design = fuel_burned_ref_design * 11.9  # in kWh
    energy_per_pax_km = energy_mission_ref_design * flights_per_fu_ref_design
    print("Energy required per FU", energy_per_pax_km)

    impact_list_ref_design = ["*", "*", "*"]
    phase_list_ref_design = ["operation", "*", "*"]
    component_list_ref_design = ["turboshaft_1", "kerosene_for_mission", "*"]

    impacts_value_ref_design = lca_impacts_search_table(
        RESULTS_SENSITIVITY_FOLDER_PATH_REF / "ref_kodiak_lca.xml_7077.xml",
        impact_list_ref_design,
        phase_list_ref_design,
        component_list_ref_design,
        rel=False,
    )

    impact_one_flight = sum(impacts_value_ref_design[:2]) / flights_per_fu_ref_design
    impact_per_kwh_of_energy_used = impact_one_flight / energy_mission_ref_design
    print("Reference design, impact per kWh", impact_per_kwh_of_energy_used)
    print("\n")

    # For the hybrid design, we consider the impacts of the fuel consumption and the production
    # of the fuel, the impact of the production of electricity for the mission and the production
    # of the battery. Last one is sanity check again, those should represent a big part of the
    # total. Results are naturally given for a kg*km, but since we want the impact per MJ of primary
    # energy used, we must first put them back to the frame of one flight and then transfer to one MJ of that flight

    hybrid_design_datafile = oad.DataFile(
        RESULTS_SENSITIVITY_FOLDER_PATH / "hybrid_kodiak_lca.xml_7077.xml"
    )
    flights_per_fu_hybrid_design = hybrid_design_datafile[
        "data:environmental_impact:flight_per_fu"
    ].value[0]
    fu_per_flights_hybrid_design = 1.0 / flights_per_fu_hybrid_design
    print("Flights per FU hybrid design", flights_per_fu_hybrid_design)
    print("FU per flights hybrid design", fu_per_flights_hybrid_design)

    single_score_hybrid_design = hybrid_design_datafile[
        "data:environmental_impact:single_score"
    ].value[0]

    # Not available here directly, have to rely on the amount of kerosene in the tanks
    fuel_burned_hybrid_design = (
        hybrid_design_datafile[
            "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_main_route"
        ].value[0]
        * 2.0
    )
    electricity_used_hybrid_design = hybrid_design_datafile[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_main_route"
    ].value[0]
    electricity_unit = hybrid_design_datafile[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_main_route"
    ].units
    if electricity_unit == "W*h":
        electricity_used_hybrid_design /= 1000.0
    fuel_burned_per_pax_km_hybrid_design = fuel_burned_hybrid_design * flights_per_fu_hybrid_design
    energy_mission_hybrid_design = (
        fuel_burned_hybrid_design * 11.9 + electricity_used_hybrid_design
    )  # in kWh
    print(
        "Energy hybridization ratio",
        electricity_used_hybrid_design / energy_mission_hybrid_design * 100.0,
    )
    energy_per_pax_km_hybrid = energy_mission_hybrid_design * flights_per_fu_hybrid_design
    print("Energy required per FU", energy_per_pax_km_hybrid)

    impact_list_hybrid_design = ["*", "*", "*", "*", "*"]
    phase_list_hybrid_design = ["operation", "*", "*", "production", "*"]
    component_list_hybrid_design = [
        "turboshaft_1",
        "kerosene_for_mission",
        "electricity_for_mission",
        "battery_pack_1",
        "*",
    ]

    impacts_value_hybrid_design = lca_impacts_search_table(
        RESULTS_SENSITIVITY_FOLDER_PATH / "hybrid_kodiak_lca.xml_7077.xml",
        impact_list_hybrid_design,
        phase_list_hybrid_design,
        component_list_hybrid_design,
        rel=False,
    )

    impact_one_flight_hybrid = sum(impacts_value_hybrid_design[:-1]) * fu_per_flights_hybrid_design
    impact_per_kwh_of_energy_used_hybrid = impact_one_flight_hybrid / energy_mission_hybrid_design
    print("Hybrid design, impact per kWh", impact_per_kwh_of_energy_used_hybrid)
    print("\n")

    print(
        "Decrease in fuel required",
        (fuel_burned_hybrid_design - fuel_burned_ref_design) / fuel_burned_ref_design * 100.0,
    )
    print(
        "Decrease in fuel required per pax.km",
        (fuel_burned_per_pax_km_hybrid_design - fuel_burned_per_pax_km_ref_design)
        / fuel_burned_per_pax_km_ref_design
        * 100.0,
    )
    print(
        "Decrease in energy required",
        (energy_mission_hybrid_design - energy_mission_ref_design)
        / energy_mission_ref_design
        * 100.0,
    )
    print(
        "Decrease in energy required per pax.km",
        (energy_per_pax_km_hybrid - energy_per_pax_km) / energy_per_pax_km * 100.0,
    )
    print(
        "Increase in environmental intensity",
        (impact_per_kwh_of_energy_used_hybrid - impact_per_kwh_of_energy_used)
        / impact_per_kwh_of_energy_used
        * 100.0,
    )

    print("Single score reference design: ", single_score_ref)
    print("Single score hybrid design: ", single_score_hybrid_design)
    print(
        "Variation in single score: ",
        (single_score_ref - single_score_hybrid_design) / single_score_ref * 100.0,
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_search_engine_paper_climate_change():
    ref_design_datafile = oad.DataFile(
        RESULTS_SENSITIVITY_FOLDER_PATH_REF / "ref_kodiak_lca.xml_7077.xml"
    )
    flights_per_fu_ref_design = ref_design_datafile[
        "data:environmental_impact:flight_per_fu"
    ].value[0]
    print("Flights per FU design", flights_per_fu_ref_design)
    print("FU per flights design", 1.0 / flights_per_fu_ref_design)

    fuel_burned_ref_design = (
        ref_design_datafile[
            "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_main_route"
        ].value[0]
        * 2.0
    )
    energy_mission_ref_design = fuel_burned_ref_design * 11.9  # in kWh

    impact_kerosene_combustion_one_fu = ref_design_datafile[
        "data:environmental_impact:climate_change:operation:turboshaft_1"
    ].value[0]
    impact_kerosene_production_one_fu = ref_design_datafile[
        "data:environmental_impact:climate_change:operation:kerosene_for_mission"
    ].value[0]

    total_impact_kerosene_one_fu = (
        impact_kerosene_combustion_one_fu + impact_kerosene_production_one_fu
    )

    impact_one_flight = total_impact_kerosene_one_fu / flights_per_fu_ref_design
    impact_per_kwh_of_energy_used = impact_one_flight / energy_mission_ref_design

    print("Kg of CO2eq for 1 kWh of kerosene", impact_per_kwh_of_energy_used)

    ################################################################################################
    # Hybrid

    hybrid_design_datafile = oad.DataFile(
        RESULTS_SENSITIVITY_FOLDER_PATH / "hybrid_kodiak_lca.xml_7077.xml"
    )
    flights_per_fu_hybrid_design = hybrid_design_datafile[
        "data:environmental_impact:flight_per_fu"
    ].value[0]
    fu_per_flights_hybrid_design = 1.0 / flights_per_fu_hybrid_design
    print("Flights per FU hybrid design", flights_per_fu_hybrid_design)
    print("FU per flights hybrid design", fu_per_flights_hybrid_design)

    electricity_used_hybrid_design = hybrid_design_datafile[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_main_route"
    ].value[0]
    electricity_unit = hybrid_design_datafile[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_main_route"
    ].units
    if electricity_unit == "W*h":
        electricity_used_hybrid_design /= 1000.0
    energy_mission_hybrid_design = electricity_used_hybrid_design  # in kWh

    impact_battery_production_one_fu = hybrid_design_datafile[
        "data:environmental_impact:climate_change:production:battery_pack_1"
    ].value[0]
    impact_electricity_production_one_fu = hybrid_design_datafile[
        "data:environmental_impact:climate_change:operation:electricity_for_mission"
    ].value[0]

    total_impact_electricity_one_fu_co2eq = (
        impact_battery_production_one_fu + impact_electricity_production_one_fu
    )
    ratio_battery_production = (
        impact_battery_production_one_fu / total_impact_electricity_one_fu_co2eq
    )
    impact_one_flight_electricity = (
        total_impact_electricity_one_fu_co2eq / flights_per_fu_hybrid_design
    )
    impact_per_kwh_of_electricity_used = (
        impact_one_flight_electricity / energy_mission_hybrid_design
    )

    print("Kg of CO2eq for 1 kWh of electricity", impact_per_kwh_of_electricity_used)
    print(
        "Kg of CO2eq for producing 1 kWh of electricity",
        (1.0 - ratio_battery_production) * impact_per_kwh_of_electricity_used,
    )
    print(
        "Kg of CO2eq for producing the battery",
        ratio_battery_production * impact_per_kwh_of_electricity_used,
    )

    impact_list_hybrid_design = ["*", "*"]
    phase_list_hybrid_design = ["*", "production"]
    component_list_hybrid_design = [
        "electricity_for_mission",
        "battery_pack_1",
    ]

    impacts_value_hybrid_design = lca_impacts_search_table(
        RESULTS_SENSITIVITY_FOLDER_PATH / "hybrid_kodiak_lca.xml_7077.xml",
        impact_list_hybrid_design,
        phase_list_hybrid_design,
        component_list_hybrid_design,
        rel=False,
    )

    total_impact_electricity_one_fu_single_score = sum(impacts_value_hybrid_design)

    impact_one_flight_electricity_single_score = (
        total_impact_electricity_one_fu_single_score / flights_per_fu_hybrid_design
    )
    impact_per_kwh_of_electricity_used_single_score = (
        impact_one_flight_electricity_single_score / energy_mission_hybrid_design
    )

    print(
        "Single score for 1 kWh of electricity",
        impact_per_kwh_of_electricity_used_single_score * 1e5,
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_absolute_phase_hybrid():
    fig = lca_impacts_bar_chart_with_components_absolute(
        RESULTS_SENSITIVITY_FOLDER_PATH / "hybrid_kodiak_lca.xml_7077.xml",
        name_aircraft="Hybrid Kodiak 100",
        detailed_component_contributions=True,
        legend_rename={
            "manufacturing": "line testing",
            "turboshaft: operation": "kerosene combustion",
            "kerosene for mission: operation": "kerosene production",
        },
        cutoff_criteria=1.0,
    )
    fig.update_layout(title_text=None)

    fig.show()
    fig.update_layout(height=800, width=1600)
    # Somehow this prevents the ugly footer from appearing !
    fig.write_image(FIGURE_FOLDER_PATH / "ga_component_contribution_ref.pdf")
    time.sleep(3)
    fig.write_image(FIGURE_FOLDER_PATH / "ga_component_contribution_ref.pdf")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_presentation():
    fig = lca_impacts_bar_chart_with_components_absolute(
        RESULTS_SENSITIVITY_FOLDER_PATH / "hybrid_kodiak_lca.xml_7077.xml",
        name_aircraft="Hybrid Kodiak 100",
        detailed_component_contributions=True,
        cutoff_criteria=2.5,
        legend_rename={
            "battery pack: production": "Battery pack production",
            "airframe: production": "Airframe production",
            "turboshaft: operation": "Kerosene combustion",
            "kerosene for mission: operation": "Kerosene production",
        },
    )
    fig.update_layout(title_text=None)
    fig.update_layout(height=800, width=1600, font=dict(size=20))
    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_search_engine_presentation():
    impact_list = ["*"]
    phase_list = ["*"]
    component_list = ["battery_pack_1"]

    impacts_value = lca_impacts_search_table(
        RESULTS_SENSITIVITY_FOLDER_PATH / "hybrid_kodiak_lca.xml_7077.xml",
        impact_list,
        phase_list,
        component_list,
        rel=True,
    )
    print(impacts_value)