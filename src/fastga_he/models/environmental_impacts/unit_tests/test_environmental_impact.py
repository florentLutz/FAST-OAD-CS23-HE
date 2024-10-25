# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os

import pathlib
import pytest

from ..simple_energy_impact import SimpleEnergyImpacts
from ..lca_core import LCACore
from ..lca_aircraft_per_fu import LCAAircraftPerFU
from ..lca_use_flight_per_fu import LCAUseFlightPerFU
from ..lca_wing_weight_per_fu import LCAWingWeightPerFU
from ..lca_fuselage_weight_per_fu import LCAFuselageWeightPerFU
from ..lca_htp_weight_per_fu import LCAHTPWeightPerFU
from ..lca_vtp_weight_per_fu import LCAVTPWeightPerFU
from ..lca_landing_gear_weight_per_fu import LCALandingGearWeightPerFU
from ..lca_flight_control_weight_per_fu import LCAFlightControlsWeightPerFU
from ..lca_empty_aircraft_weight_per_fu import LCAEmptyAircraftWeightPerFU
from ..lca import LCA

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "data.xml"
DATA_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "results"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_impact_sizing_jet_fuel():
    """Tests propulsion weight computation from sample XML data."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(SimpleEnergyImpacts(mission="design")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SimpleEnergyImpacts(mission="design"), ivc)
    assert problem.get_val(
        "data:environmental_impact:sizing:fuel_emissions", units="kg"
    ) == pytest.approx(3558.98, abs=1e-2)
    # The value correspond to the sizing mission of the K100 with approx 6 pax and 1100 nm of range
    # it equates to around 300gCO2/PAX/km
    assert problem.get_val(
        "data:environmental_impact:sizing:energy_emissions", units="kg"
    ) == pytest.approx(0.00, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:sizing:fuel_emissions", units="kg"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:sizing:emissions", units="kg"), abs=1e-2
    )
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        2.72, abs=1e-2
    )

    problem.check_partials(compact_print=True)


def test_impact_operational_biofuel():
    """Tests propulsion weight computation from sample XML data."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SimpleEnergyImpacts(mission="operational", fuel_type="biofuel_ft_pathway")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SimpleEnergyImpacts(mission="operational", fuel_type="biofuel_ft_pathway"), ivc
    )
    assert problem.get_val(
        "data:environmental_impact:operational:fuel_emissions", units="kg"
    ) == pytest.approx(74.37, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:energy_emissions", units="kg"
    ) == pytest.approx(41.8752, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:emissions", units="kg"
    ) == pytest.approx(116.248, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(0.275, abs=1e-2)

    problem.check_partials(compact_print=True)


def test_impact_both():
    """Tests propulsion weight computation from sample XML data."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SimpleEnergyImpacts(
                mission="both", fuel_type="biofuel_hefa_pathway", electricity_mix="france"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SimpleEnergyImpacts(
            mission="both", fuel_type="biofuel_hefa_pathway", electricity_mix="france"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:environmental_impact:sizing:fuel_emissions", units="kg"
    ) == pytest.approx(866.67, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:sizing:energy_emissions", units="kg"
    ) == pytest.approx(0.00, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(866.67, abs=1e-2)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        0.66, abs=1e-2
    )

    assert problem.get_val(
        "data:environmental_impact:operational:fuel_emissions", units="kg"
    ) == pytest.approx(208.63, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:energy_emissions", units="kg"
    ) == pytest.approx(17.05, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:emissions", units="kg"
    ) == pytest.approx(225.68, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(0.5344, abs=1e-2)

    problem.check_partials(compact_print=True)


def test_lca_without_fuel_burn():
    ivc = get_indep_var_comp(
        list_inputs(
            LCACore(
                power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
                component_level_breakdown=True,
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCACore(
            power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
            component_level_breakdown=True,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:sum"
    ) == pytest.approx(4706.81, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:climate_change:production:propeller_1"
    ) == pytest.approx(1618.39, abs=1e-2)

    # Sanity check
    assert problem.get_val(
        "data:environmental_impact:climate_change:production:sum"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:climate_change:production:propeller_1")
        + problem.get_val("data:environmental_impact:climate_change:production:motor_1")
        + problem.get_val("data:environmental_impact:climate_change:production:inverter_1")
        + problem.get_val("data:environmental_impact:climate_change:production:dc_bus_1")
        + problem.get_val("data:environmental_impact:climate_change:production:harness_1")
        + problem.get_val("data:environmental_impact:climate_change:production:dc_splitter_1")
        + problem.get_val("data:environmental_impact:climate_change:production:dc_sspc_1")
        + problem.get_val("data:environmental_impact:climate_change:production:dc_sspc_2")
        + problem.get_val("data:environmental_impact:climate_change:production:battery_pack_1")
        + problem.get_val("data:environmental_impact:climate_change:production:battery_pack_2"),
        abs=1e-4,
    )

    problem.check_partials(compact_print=True)


def test_aircraft_per_fu_pax_km():
    inputs_list = [
        "data:TLAR:aircraft_lifespan",
        "data:TLAR:flight_per_year",
        "data:TLAR:range",
        "data:weight:aircraft:payload",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAAircraftPerFU(),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:aircraft_per_fu") == pytest.approx(
        1.70306211e-06, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_use_flight_per_fu_pax_km():
    inputs_list = [
        "data:TLAR:range",
        "data:weight:aircraft:payload",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAUseFlightPerFU(),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:flight_per_fu") == pytest.approx(
        0.00932427, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_wing_weight_per_fu():
    inputs_list = [
        "data:environmental_impact:aircraft_per_fu",
        "data:weight:airframe:wing:mass",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAWingWeightPerFU(),
        ivc,
    )

    assert problem.get_val("data:weight:airframe:wing:mass_per_fu", units="kg") == pytest.approx(
        9.07426865e-05, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_fuselage_weight_per_fu():
    inputs_list = [
        "data:environmental_impact:aircraft_per_fu",
        "data:weight:airframe:fuselage:mass",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAFuselageWeightPerFU(),
        ivc,
    )

    assert problem.get_val(
        "data:weight:airframe:fuselage:mass_per_fu", units="kg"
    ) == pytest.approx(5.99973828e-05, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_htp_weight_per_fu():
    inputs_list = [
        "data:environmental_impact:aircraft_per_fu",
        "data:weight:airframe:horizontal_tail:mass",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAHTPWeightPerFU(),
        ivc,
    )

    assert problem.get_val(
        "data:weight:airframe:horizontal_tail:mass_per_fu", units="kg"
    ) == pytest.approx(2.98789384e-06, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_vtp_weight_per_fu():
    inputs_list = [
        "data:environmental_impact:aircraft_per_fu",
        "data:weight:airframe:vertical_tail:mass",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAVTPWeightPerFU(),
        ivc,
    )

    assert problem.get_val(
        "data:weight:airframe:vertical_tail:mass_per_fu", units="kg"
    ) == pytest.approx(3.45032812e-06, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_lg_weight_per_fu():
    inputs_list = [
        "data:environmental_impact:aircraft_per_fu",
        "data:weight:airframe:landing_gear:main:mass",
        "data:weight:airframe:landing_gear:front:mass",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCALandingGearWeightPerFU(),
        ivc,
    )

    assert problem.get_val(
        "data:weight:airframe:landing_gear:mass_per_fu", units="kg"
    ) == pytest.approx(1.87042076e-05, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_flight_controls_weight_per_fu():
    inputs_list = [
        "data:environmental_impact:aircraft_per_fu",
        "data:weight:airframe:flight_controls:mass",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAFlightControlsWeightPerFU(),
        ivc,
    )

    assert problem.get_val(
        "data:weight:airframe:flight_controls:mass_per_fu", units="kg"
    ) == pytest.approx(8.25877812e-06, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_empty_aircraft_weight_per_fu():
    inputs_list = [
        "data:environmental_impact:aircraft_per_fu",
        "data:weight:aircraft:OWE",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAEmptyAircraftWeightPerFU(),
        ivc,
    )

    assert problem.get_val("data:weight:aircraft:OWE_per_fu", units="kg") == pytest.approx(
        0.00042658, rel=1e-3
    )

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel():
    ivc = get_indep_var_comp(
        list_inputs(
            LCA(
                power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
                component_level_breakdown=True,
                airframe_material="composite",
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCA(
            power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
            component_level_breakdown=True,
            airframe_material="composite",
        ),
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_lca.xml"
    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:propeller_1"
    ) == pytest.approx(0.00528819, rel=1e-4)
    assert problem.get_val(
        "data:environmental_impact:total_natural_resources:production:propeller_1"
    ) == pytest.approx(0.00030415567672208426, rel=1e-4)

    assert problem.get_val("data:environmental_impact:aircraft_per_fu") == pytest.approx(
        1.70306211e-06, rel=1e-2
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass_per_fu"
    ) == pytest.approx(0.00140503, rel=1e-2)

    # Sanity check
    assert problem.get_val(
        "data:environmental_impact:climate_change:production:sum"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:climate_change:production:propeller_1")
        + problem.get_val("data:environmental_impact:climate_change:production:motor_1")
        + problem.get_val("data:environmental_impact:climate_change:production:inverter_1")
        + problem.get_val("data:environmental_impact:climate_change:production:dc_bus_1")
        + problem.get_val("data:environmental_impact:climate_change:production:harness_1")
        + problem.get_val("data:environmental_impact:climate_change:production:dc_splitter_1")
        + problem.get_val("data:environmental_impact:climate_change:production:dc_sspc_1")
        + problem.get_val("data:environmental_impact:climate_change:production:dc_sspc_2")
        + problem.get_val("data:environmental_impact:climate_change:production:battery_pack_1")
        + problem.get_val("data:environmental_impact:climate_change:production:battery_pack_2")
        + problem.get_val("data:environmental_impact:climate_change:production:wing")
        + problem.get_val("data:environmental_impact:climate_change:production:fuselage")
        + problem.get_val("data:environmental_impact:climate_change:production:horizontal_tail")
        + problem.get_val("data:environmental_impact:climate_change:production:vertical_tail")
        + problem.get_val("data:environmental_impact:climate_change:production:landing_gear")
        + problem.get_val("data:environmental_impact:climate_change:production:flight_controls")
        + problem.get_val("data:environmental_impact:climate_change:production:assembly"),
        rel=1e-4,
    )

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:sum"
    ) == pytest.approx(0.11821691, rel=1e-5)

    assert problem.get_val(
        "data:environmental_impact:climate_change:operation:battery_pack_1"
    ) == pytest.approx(0.0, abs=1e-5)

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_tbm900():
    ivc = get_indep_var_comp(
        list_inputs(
            LCA(
                power_train_file_path=DATA_FOLDER_PATH / "tbm900_propulsion.yml",
                component_level_breakdown=True,
                airframe_material="aluminium",
            )
        ),
        __file__,
        DATA_FOLDER_PATH / "tbm900.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCA(
            power_train_file_path=DATA_FOLDER_PATH / "tbm900_propulsion.yml",
            component_level_breakdown=True,
            airframe_material="aluminium",
        ),
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "tbm900_lca.xml"
    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:sum"
    ) == pytest.approx(0.00241945, rel=1e-5)

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:turboshaft_1"
    ) == pytest.approx(0.00034762, rel=1e-3)

    assert problem.get_val(
        "data:environmental_impact:climate_change:operation:turboshaft_1"
    ) == pytest.approx(0.21645, rel=1e-3)

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_cirrus_sr22():
    ivc = get_indep_var_comp(
        list_inputs(
            LCA(
                power_train_file_path=DATA_FOLDER_PATH / "sr22_propulsion.yml",
                component_level_breakdown=True,
                airframe_material="composite",
            )
        ),
        __file__,
        DATA_FOLDER_PATH / "sr22.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCA(
            power_train_file_path=DATA_FOLDER_PATH / "sr22_propulsion.yml",
            component_level_breakdown=True,
            airframe_material="composite",
        ),
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "sr22_lca.xml"
    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:human_toxicity_non-carcinogenic:operation:ice_1"
    ) == pytest.approx(0.69629354, rel=1e-3)
    assert problem.get_val(
        "data:environmental_impact:human_toxicity_carcinogenic:operation:ice_1"
    ) == pytest.approx(0.00058180, rel=1e-3)

    problem.check_partials(compact_print=True)
