# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth
from shutil import rmtree
import logging
import pytest
import numpy as np
import openmdao.api as om
import fastoad.api as oad
import plotly.graph_objects as go

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
WORKDIR_FOLDER_PATH = pth.join(pth.dirname(__file__), "workdir")

RESULTS_SENSITIVITY_FOLDER_PATH = pth.join(pth.dirname(__file__), "results_sensitivity")
RESULTS_FULL_SIZING_SENSITIVITY_FOLDER_PATH = pth.join(
    pth.dirname(__file__), "results_sensitivity_full_sizing"
)
RESULTS_FULL_SIZING_SENSITIVITY_FOLDER_PATH_2 = pth.join(
    pth.dirname(__file__), "results_sensitivity_full_sizing_2"
)
RESULTS_FULL_SIZING_SENSITIVITY_FOLDER_PATH_3 = pth.join(
    pth.dirname(__file__), "results_sensitivity_full_sizing_3"
)


@pytest.fixture(scope="module")
def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree(WORKDIR_FOLDER_PATH, ignore_errors=True)


def test_sizing_kodiak_100():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_kodiak100.xml"
    process_file_name = "full_sizing_kodiak100.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_kodiak100.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3280.0, rel=1e-2
    )
    # Actual value is 3290 kg
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        1727.0, rel=1e-2
    )
    # Actual value is 1712 kg
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        933.00, rel=1e-2
    )
    # Actual value is 2110 lbs or 960 kg


def test_sizing_kodiak_100_full_electric():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_elec_kodiak100.xml"
    process_file_name = "full_sizing_kodiak100_elec.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_kodiak100_elec.html")

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 5.0,
        "cell_weight_ref": 45.0e-3,
    }

    problem.setup()

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        3394.82, rel=1e-2
    )
    # Actual value is 3290 kg
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(
        2842.82, rel=1e-2
    )


def test_operational_mission_kodiak_100():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_kodiak100_op_mission.xml"
    process_file_name = "operational_mission_kodiak100.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:mission:operational:TOW", units="kg") == pytest.approx(
        3113.0, abs=1
    )
    assert problem.get_val("data:mission:operational:fuel", units="kg") == pytest.approx(
        246.00, abs=1
    )
    assert problem.get_val(
        "data:environmental_impact:operational:fuel_emissions", units="kg"
    ) == pytest.approx(938.0, abs=1)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(2.223, abs=1e-2)


def test_retrofit_hybrid_kodiak():
    """

    We'll take a new turboshaft that correspond to the PW206B as it seems to have a fairly good
    sfc according to https://en.wikipedia.org/wiki/Pratt_%26_Whitney_Canada_PW200. We'll use that
    reference sfc as well as some educated on OPR and thermodynamic to get the right k_sfc before
    we can get our hand on more data (possibly from Jane's). We'll consider that sfc is given at
    Sea Level Static with power equal to limit power. This gives an k_sfc of 1.11

    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_kodiak.xml"
    process_file_name = "hybrid_kodiak_retrofit.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

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

    # om.n2(problem)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(210.00, abs=1.0)
    assert problem.get_val("data:propulsion:he_power_train:mass", units="kg") == pytest.approx(
        529.0, abs=1.0
    )
    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(805.00, abs=1.0)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        1.924, abs=1e-2
    )


def test_retrofit_hybrid_kodiak_with_lca_varying_battery_lifespan():
    """

    We'll take a new turboshaft that correspond to the PW206B as it seems to have a fairly good
    sfc according to https://en.wikipedia.org/wiki/Pratt_%26_Whitney_Canada_PW200. We'll use that
    reference sfc as well as some educated on OPR and thermodynamic to get the right k_sfc before
    we can get our hand on more data (possibly from Jane's). We'll consider that sfc is given at
    Sea Level Static with power equal to limit power. This gives an k_sfc of 1.11

    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "input_hybrid_kodiak_with_lca.xml"
    process_file_name = "hybrid_kodiak_retrofit_with_lca.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

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

    # om.n2(problem)

    problem.run_model()

    soc_start_array = np.linspace(100.0, 60.0, 20)
    for soc_start in soc_start_array:
        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start",
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
    flights_per_fu = []

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
        flights_per_fu.append(datafile["data:environmental_impact:flight_per_fu"].value[0])

    sorted_single_scores = [single_scores[i] for i in np.argsort(socs_start_mission)]
    sorted_battery_mass_per_fu = [battery_mass_per_fu[i] for i in np.argsort(socs_start_mission)]
    flights_per_fu_sorted = [flights_per_fu[i] for i in np.argsort(socs_start_mission)]
    sorted_socs_start = np.sort(socs_start_mission)

    sorted_generic_list = [generic_list[i] for i in np.argsort(socs_start_mission)]
    sorted_battery_mass = [battery_mass[i] for i in np.argsort(socs_start_mission)]

    scatter = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_single_scores) / sorted_single_scores[-1],
        mode="lines+markers",
        name="Single score variation",
        marker=dict(symbol="circle", size=10),
    )
    fig.add_trace(scatter)
    scatter_2 = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_battery_mass_per_fu) / sorted_battery_mass_per_fu[-1],
        mode="lines+markers",
        name="Battery mass per FU variation",
        marker=dict(symbol="square", size=10),
    )
    fig.add_trace(scatter_2)
    scatter_3 = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_generic_list) / sorted_generic_list[-1],
        mode="lines+markers",
        name="Battery lifespan variation",
        marker=dict(symbol="diamond", size=10),
    )
    fig.add_trace(scatter_3)
    scatter_4 = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_battery_mass) / sorted_battery_mass[-1],
        mode="lines+markers",
        name="Battery mass variation",
        marker=dict(symbol="cross", size=10),
    )
    fig.add_trace(scatter_4)
    scatter_5 = go.Scatter(
        x=sorted_socs_start,
        y=flights_per_fu_sorted[-1] / np.array(flights_per_fu_sorted),
        mode="lines+markers",
        name="Functional unit per flight variation",
        marker=dict(symbol="x", size=10),
    )
    fig.add_trace(scatter_5)

    fig.update_layout(
        title_text="Evolution of quantity of interest for the retrofit Kodiak case",
        title_x=0.5,
        xaxis_title="Initial SoC [%]",
        height=800,
        width=1600,
        font_size=18,
    )
    fig.update_yaxes(title="Variation with respect to 100% SOC case")

    fig.show()


def test_full_sizing_hybrid_kodiak_100():
    """Test the overall aircraft design process with wing positioning on the hybrid K100."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "input_full_sizing_hybrid_kodiak.xml"
    process_file_name = "full_sizing_hybrid_kodiak.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # Change battery pack characteristics so that they match those of a high power,
    # lower capacity cell like the Samsung INR18650-25R, we also take the weight fraction of the
    # Pipistrel battery. Assumes same polarization curve. And we'll take the same aging model
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

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=30.0
    )
    problem.set_val("data:weight:aircraft:MTOW", val=3000.0, units="kg")
    problem.set_val("data:geometry:wing:area", val=22.5, units="m**2")

    problem.run_model()

    # Register the baseline case
    problem.write_outputs()

    soc_start_array = np.linspace(100.0, 60.0, 20)
    for soc_start in soc_start_array:
        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start",
            units="percent",
            val=soc_start,
        )

        problem.run_model()

        problem.output_file_path = pth.join(
            RESULTS_FULL_SIZING_SENSITIVITY_FOLDER_PATH, str(int(soc_start)) + "_soc_start_out.xml"
        )
        problem.write_outputs()


def test_full_sizing_hybrid_kodiak_100_20_percent_renewal():
    """Test the overall aircraft design process with wing positioning on the hybrid K100."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "input_full_sizing_hybrid_kodiak.xml"
    process_file_name = "full_sizing_hybrid_kodiak.yml"

    # if you want to see the value for which the battery grows quicker than the lifespan increases
    # inspect between 100 and 80
    soc_start_array = np.linspace(100.0, 90.0, 11)
    for soc_start in soc_start_array:
        configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
        problem = configurator.get_problem()

        # Create inputs
        ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
        # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

        problem.write_needed_inputs(ref_inputs)
        problem.read_inputs()

        # Change battery pack characteristics so that they match those of a high power,
        # lower capacity cell like the Samsung INR18650-25R, we also take the weight fraction of the
        # Pipistrel battery. Assumes same polarization curve. And we'll take the same aging model
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

        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=30.0
        )
        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:aging:cyclic_effect_k_factor",
            val=1.0,
        )
        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:end_of_life_relative_capacity_loss",
            val=0.2,
            units="unitless",
        )
        problem.set_val("data:weight:aircraft:MTOW", val=3000.0, units="kg")
        problem.set_val("data:geometry:wing:area", val=22.5, units="m**2")

        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start",
            units="percent",
            val=soc_start,
        )

        problem.run_model()

        problem.output_file_path = pth.join(
            RESULTS_FULL_SIZING_SENSITIVITY_FOLDER_PATH_2,
            str(int(soc_start)) + "_soc_start_out.xml",
        )
        problem.write_outputs()


def test_full_sizing_hybrid_kodiak_100_20_percent_renewal_bug_question_mark():
    """Test the overall aircraft design process with wing positioning on the hybrid K100."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True
    logging.getLogger("bw2data").disabled = True
    logging.getLogger("bw2calc").disabled = True

    # Define used files depending on options
    xml_file_name = "input_full_sizing_hybrid_kodiak.xml"
    process_file_name = "full_sizing_hybrid_kodiak.yml"

    # if you want to see the value for which the battery grows quicker than the lifespan increases
    # inspect between 100 and 80

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # Change battery pack characteristics so that they match those of a high power,
    # lower capacity cell like the Samsung INR18650-25R, we also take the weight fraction of the
    # Pipistrel battery. Assumes same polarization curve. And we'll take the same aging model
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

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=30.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:end_of_life_relative_capacity_loss",
        val=0.2,
        units="unitless",
    )
    # For the """bugged""" results, turn rtol down to 1e-3 and comment the following line
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:aging:cyclic_effect_k_factor",
        val=1.0,
    )
    problem.set_val("data:weight:aircraft:MTOW", val=3000.0, units="kg")
    problem.set_val("data:geometry:wing:area", val=22.5, units="m**2")

    soc_start_array = np.linspace(100.0, 90.0, 11)
    for soc_start in soc_start_array:
        problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start",
            units="percent",
            val=soc_start,
        )

        problem.run_model()

        problem.output_file_path = pth.join(
            RESULTS_FULL_SIZING_SENSITIVITY_FOLDER_PATH_3,
            str(int(soc_start)) + "_soc_start_out_bug.xml",
        )
        problem.write_outputs()


def test_post_process_results_full_sizing():
    fig = go.Figure()

    socs_start_mission = []
    single_scores = []
    battery_mass = []

    generic_list = []

    sensitivity_study = RESULTS_FULL_SIZING_SENSITIVITY_FOLDER_PATH_3
    # or RESULTS_FULL_SIZING_SENSITIVITY_FOLDER_PATH
    # or RESULTS_FULL_SIZING_SENSITIVITY_FOLDER_PATH_3

    for file in os.listdir(sensitivity_study):
        datafile = oad.DataFile(pth.join(sensitivity_study, file))
        socs_start_mission.append(
            datafile[
                "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_mission_start"
            ].value[0]
        )
        single_scores.append(datafile["data:environmental_impact:single_score"].value[0])
        battery_mass.append(
            datafile["data:propulsion:he_power_train:battery_pack:battery_pack_1:mass"].value[0]
        )
        generic_list.append(
            datafile["data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan"].value[0]
        )

    sorted_single_scores = [single_scores[i] for i in np.argsort(socs_start_mission)]
    sorted_socs_start = np.sort(socs_start_mission)

    sorted_generic_list = [generic_list[i] for i in np.argsort(socs_start_mission)]
    sorted_battery_mass = [battery_mass[i] for i in np.argsort(socs_start_mission)]

    scatter = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_single_scores) / sorted_single_scores[-1],
        mode="lines+markers",
        name="Single score variation",
        marker=dict(symbol="circle", size=10),
    )
    fig.add_trace(scatter)
    scatter_3 = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_generic_list) / sorted_generic_list[-1],
        mode="lines+markers",
        name="Battery lifespan variation",
        marker=dict(symbol="diamond", size=10),
    )
    fig.add_trace(scatter_3)
    scatter_4 = go.Scatter(
        x=sorted_socs_start,
        y=np.array(sorted_battery_mass) / sorted_battery_mass[-1],
        mode="lines+markers",
        name="Battery mass variation",
        marker=dict(symbol="cross", size=10),
    )
    fig.add_trace(scatter_4)

    fig.update_layout(
        title_text="Evolution of quantity of interest for the fully sized hybrid Kodiak case",
        title_x=0.5,
        xaxis_title="Initial SoC [%]",
        height=800,
        width=1600,
        font_size=18,
    )
    fig.update_yaxes(title="Variation with respect to 100% SOC case")

    fig.show()


def test_retrofit_hybrid_kodiak_european_mix():
    """
    Computation of the emissions factor with the Europe electricity index.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "hybrid_kodiak_full_sizing.xml"
    process_file_name = "hybrid_kodiak_emissions_europe_mix.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(809.405, abs=1.0)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        1.933, abs=1e-2
    )


def test_retrofit_hybrid_kodiak_eu_mix_ft():
    """

    Computation of the emissions factor with the french electricity emission index and biofuel
    obtained with FT pathway.

    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "hybrid_kodiak_full_sizing.xml"
    process_file_name = "hybrid_kodiak_emissions_europe_mix_ft.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(76.19, rel=1e-3)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        0.182, rel=1e-3
    )


def test_operational_mission_kodiak_100_ft():
    """
    Computation of the emissions factor with the french electricity emission index and biofuel
    obtained with FT pathway.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "op_mission_full.xml"
    process_file_name = "op_kodiak_emissions_ft.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:operational:emissions", units="kg"
    ) == pytest.approx(81.505, abs=1.0)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(0.1930, abs=1e-2)


def test_retrofit_hybrid_kodiak_eu_mix_hefa():
    """

    Computation of the emissions factor with the french electricity emission index and biofuel
    obtained with HEFA pathway.

    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "hybrid_kodiak_full_sizing.xml"
    process_file_name = "hybrid_kodiak_emissions_europe_mix_hefa.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(202.01, rel=1e-3)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        0.4826, rel=1e-3
    )


def test_operational_mission_kodiak_100_hefa():
    """
    Computation of the emissions factor with the french electricity emission index and biofuel
    obtained with HEFA pathway.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "op_mission_full.xml"
    process_file_name = "op_kodiak_emissions_hefa.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    problem.run_model()

    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:operational:emissions", units="kg"
    ) == pytest.approx(228.638, abs=1.0)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(0.5415, abs=1e-2)
