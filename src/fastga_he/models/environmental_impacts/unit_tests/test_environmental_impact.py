# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import pathlib

import pytest

import fastoad.api as oad

from lca_modeller.gui.plots import process_tree

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from ..lca import LCA
from ..lca_equivalent_year_of_life import LCAEquivalentYearOfLife
from ..lca_equivalent_flight_per_year import LCAEquivalentFlightsPerYear
from ..lca_max_airframe_hours import LCAEquivalentMaxAirframeHours
from ..lca_aircraft_per_fu import LCAAircraftPerFU, LCAAircraftPerFUFlightHours
from ..lca_delivery_mission_ratio import LCARatioDeliveryFlightMission
from ..lca_distribution_cargo import LCADistributionCargoMassDistancePerFU
from ..lca_electricty_per_fu import LCAElectricityPerFU, LCAElectricityPerFUFromUsePhaseValue
from ..lca_empty_aircraft_weight_per_fu import LCAEmptyAircraftWeightPerFU
from ..lca_flight_control_weight_per_fu import LCAFlightControlsWeightPerFU
from ..lca_fuselage_weight_per_fu import LCAFuselageWeightPerFU
from ..lca_gasoline_per_fu import LCAGasolinePerFU
from ..lca_htp_weight_per_fu import LCAHTPWeightPerFU
from ..lca_kerosene_per_fu import LCAKerosenePerFU
from ..lca_landing_gear_weight_per_fu import LCALandingGearWeightPerFU
from ..lca_line_test_mission_ratio import LCARatioTestFlightMission
from ..lca_use_flight_per_fu import LCAUseFlightPerFU, LCAUseFlightPerFUFlightHours
from ..lca_vtp_weight_per_fu import LCAVTPWeightPerFU
from ..lca_wing_weight_per_fu import LCAWingWeightPerFU
from ..simple_energy_impact import SimpleEnergyImpacts

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
        390.41, abs=1e-2
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
    ) == pytest.approx(2.324, abs=1e-2)

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
        95.07, abs=1e-2
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
    ) == pytest.approx(4.513, abs=1e-2)

    problem.check_partials(compact_print=True)


def test_aircraft_equivalent_year_of_life():
    inputs_list = [
        "data:TLAR:max_airframe_hours",
        "data:TLAR:flight_hours_per_year",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAEquivalentYearOfLife(),
        ivc,
    )

    assert problem.get_val("data:TLAR:aircraft_lifespan", units="yr") == pytest.approx(
        12.446, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_aircraft_equivalent_flight_per_year():
    inputs_list = [
        "data:TLAR:max_airframe_hours",
        "data:mission:sizing:main_route:duration",
        "data:mission:operational:main_route:duration",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAEquivalentFlightsPerYear(use_operational_mission=False),
        ivc,
    )

    assert problem.get_val("data:TLAR:flight_per_year") == pytest.approx(291.762, rel=1e-3)

    problem.check_partials(compact_print=True)

    problem = run_system(
        LCAEquivalentFlightsPerYear(use_operational_mission=True),
        ivc,
    )

    assert problem.get_val("data:TLAR:flight_per_year") == pytest.approx(72.94, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_equivalent_max_airframe_hours():
    inputs_list = [
        "data:TLAR:aircraft_lifespan",
        "data:TLAR:flight_per_year",
        "data:mission:sizing:main_route:duration",
        "data:mission:operational:main_route:duration",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAEquivalentMaxAirframeHours(),
        ivc,
    )

    assert problem.get_val("data:TLAR:max_airframe_hours", units="h") == pytest.approx(
        5314.34, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    problem = run_system(
        LCAEquivalentMaxAirframeHours(use_operational_mission=True),
        ivc,
    )

    assert problem.get_val("data:TLAR:max_airframe_hours", units="h") == pytest.approx(
        21256.68, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_aircraft_per_fu_pax_km():
    inputs_list = [
        "data:TLAR:aircraft_lifespan",
        "data:TLAR:flight_per_year",
        "data:TLAR:range",
        "data:mission:operational:range",
        "data:weight:aircraft:payload",
        "data:mission:operational:payload:mass",
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

    problem = run_system(
        LCAAircraftPerFU(use_operational_mission=True),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:aircraft_per_fu") == pytest.approx(
        3.10502283e-07, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_aircraft_per_fu_flight_hours():
    inputs_list = [
        "data:TLAR:aircraft_lifespan",
        "data:TLAR:flight_per_year",
        "data:mission:sizing:main_route:duration",
        "data:mission:operational:main_route:duration",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAAircraftPerFUFlightHours(),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:aircraft_per_fu") == pytest.approx(
        0.00018817, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    problem = run_system(
        LCAAircraftPerFUFlightHours(use_operational_mission=True),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:aircraft_per_fu") == pytest.approx(
        4.70440185e-05, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_use_flight_per_fu_pax_km():
    inputs_list = [
        "data:TLAR:range",
        "data:mission:operational:range",
        "data:weight:aircraft:payload",
        "data:mission:operational:payload:mass",
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

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAUseFlightPerFU(use_operational_mission=True),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:flight_per_fu") == pytest.approx(
        0.0017, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_use_flight_per_fu_flight_hours():
    inputs_list = [
        "data:mission:sizing:main_route:duration",
        "data:mission:operational:main_route:duration",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAUseFlightPerFUFlightHours(),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:flight_per_fu") == pytest.approx(
        1.03023633, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAUseFlightPerFUFlightHours(use_operational_mission=True),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:flight_per_fu") == pytest.approx(
        0.257566, rel=1e-3
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

    # Check it still works with the option enabled
    problem = run_system(
        LCAWingWeightPerFU(airframe_material="composite"),
        ivc,
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

    # Check that it still works with the option enabled
    problem = run_system(
        LCAFuselageWeightPerFU(airframe_material="composite"),
        ivc,
    )

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

    # Check that it still works with the option enabled
    problem = run_system(
        LCAHTPWeightPerFU(airframe_material="composite"),
        ivc,
    )

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

    # Check that it still works with the option enabled
    problem = run_system(
        LCAVTPWeightPerFU(airframe_material="composite"),
        ivc,
    )

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


def test_line_tests_sizing_ratio():
    inputs_list = [
        "data:mission:sizing:main_route:duration",
        "data:mission:operational:main_route:duration",
        "data:environmental_impact:line_test:duration",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCARatioTestFlightMission(),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:line_test:mission_ratio") == pytest.approx(
        10.30236332, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCARatioTestFlightMission(use_operational_mission=True),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:line_test:mission_ratio") == pytest.approx(
        2.57566001, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_delivery_sizing_ratio():
    inputs_list = [
        "data:TLAR:range",
        "data:environmental_impact:delivery:distance",
        "data:mission:operational:range",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCARatioDeliveryFlightMission(),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:delivery:mission_ratio") == pytest.approx(
        150.94, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCARatioDeliveryFlightMission(use_operational_mission=True),
        ivc,
    )

    assert problem.get_val("data:environmental_impact:delivery:mission_ratio") == pytest.approx(
        32.0, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_delivery_ton_km():
    inputs_list = [
        "data:weight:aircraft:OWE",
        "data:environmental_impact:delivery:distance",
        "data:environmental_impact:aircraft_per_fu",
    ]

    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCADistributionCargoMassDistancePerFU(),
        ivc,
    )

    assert problem.get_val(
        "data:environmental_impact:delivery:cargo_transport_per_fu"
    ) == pytest.approx(0.00341267, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_electricity_per_fu_velis():
    inputs_list = [
        "data:environmental_impact:flight_per_fu",
        "data:environmental_impact:aircraft_per_fu",
        "data:environmental_impact:line_test:mission_ratio",
        "data:environmental_impact:delivery:mission_ratio",
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_main_route",
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:energy_consumed_main_route",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, DATA_FOLDER_PATH / "data.xml")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAElectricityPerFU(
            batteries_name_list=["battery_pack_1", "battery_pack_2"],
            batteries_type_list=["battery_pack", "battery_pack"],
        ),
        ivc,
    )

    assert problem.get_val(
        "data:LCA:operation:he_power_train:electricity:energy_per_fu", units="W*h"
    ) == pytest.approx(233.23046823, rel=1e-5)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:electricity:energy_per_fu", units="W*h"
    ) == pytest.approx(0.25769592, rel=1e-5)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:electricity:energy_per_fu", units="W*h"
    ) == pytest.approx(1.25066409, rel=1e-5)

    problem.check_partials(compact_print=True)


def test_electricity_per_fu_velis_from_use_phase_value():
    inputs_list = [
        "data:environmental_impact:flight_per_fu",
        "data:environmental_impact:aircraft_per_fu",
        "data:environmental_impact:line_test:mission_ratio",
        "data:environmental_impact:delivery:mission_ratio",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, DATA_FOLDER_PATH / "data.xml")
    ivc.add_output(
        "data:LCA:operation:he_power_train:electricity:energy_per_fu", val=233.23046823, units="W*h"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAElectricityPerFUFromUsePhaseValue(
            batteries_name_list=["battery_pack_1", "battery_pack_2"],
            batteries_type_list=["battery_pack", "battery_pack"],
        ),
        ivc,
    )
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:electricity:energy_per_fu", units="W*h"
    ) == pytest.approx(0.25769592, rel=1e-5)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:electricity:energy_per_fu", units="W*h"
    ) == pytest.approx(1.25066409, rel=1e-5)

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel():
    ivc = get_indep_var_comp(
        list_inputs(
            LCA(
                power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
                component_level_breakdown=True,
                airframe_material="composite",
                delivery_method="train",
                electric_mix="french",
                normalization=True,
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
            delivery_method="train",
            electric_mix="french",
            normalization=True,
        ),
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_lca.xml"
    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:propeller_1"
    ) == pytest.approx(0.0027558696608642936, rel=1e-4)
    assert problem.get_val(
        "data:environmental_impact:total_natural_resources:production:propeller_1"
    ) == pytest.approx(0.0001303969218601463, rel=1e-4)

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
    ) == pytest.approx(0.07661772257755672, rel=1e-5)

    assert problem.get_val(
        "data:environmental_impact:climate_change:operation:battery_pack_1"
    ) == pytest.approx(0.0, abs=1e-5)

    assert problem.get_val(
        "data:environmental_impact:climate_change:operation:sum"
    ) == pytest.approx(0.015348882271229677, abs=1e-5)

    assert problem.get_val(
        "data:environmental_impact:climate_change:manufacturing:sum"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:line_test:mission_ratio")
        * problem.get_val("data:environmental_impact:climate_change:operation:sum")
        * problem.get_val("data:environmental_impact:aircraft_per_fu")
        / problem.get_val("data:environmental_impact:flight_per_fu"),
        abs=1e-5,
    )

    assert problem.get_val(
        "data:environmental_impact:climate_change:distribution:sum"
    ) == pytest.approx(0.00027046063751871727, rel=1e-4)

    problem.check_partials(compact_print=True)

    lca_model = problem.model.component.lca_core.model

    process_tree(lca_model, outfile=os.path.join(RESULTS_FOLDER_PATH, "lca_pipistrel.html"))


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_pipistrel_lca_comparison_paper():
    # The analysis we try to replicate here is not exactly the pipistrel we size and test earlier.
    # Rather it looks to be an earlier version of the pipistrel whose empty weight is 370kg as
    # opposed to the 428kg of the Valis Electro. We will, however, try to replicate those result
    # by adjusting the content of the LCA conf file. We'll make the assumption of a similar
    # powertrain (but different values).

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        normalization=True,
        weighting=True,
        aircraft_lifespan_in_hours=True,
        ecoinvent_version="3.9.1",
        functional_unit="Flight hours",
        write_lca_conf=False,
        lca_conf_file_path=DATA_FOLDER_PATH / "pipistrel_alpha_electro.yml",
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        "pipistrel_alpha.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_alpha_standard.xml"
    problem.write_outputs()

    problem.set_val("data:TLAR:max_airframe_hours", units="h", val=500.0)
    problem.run_model()

    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_alpha_short.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_extract_results_from_pipistrel_lca_comparison_standard():
    alpha_standard_datafile = oad.DataFile(RESULTS_FOLDER_PATH / "pipistrel_alpha_standard.xml")

    airframe_impact = (
        alpha_standard_datafile[
            "data:environmental_impact:climate_change:production:flight_controls"
        ].value[0]
        + alpha_standard_datafile[
            "data:environmental_impact:climate_change:production:fuselage"
        ].value[0]
        + alpha_standard_datafile[
            "data:environmental_impact:climate_change:production:horizontal_tail"
        ].value[0]
        + alpha_standard_datafile[
            "data:environmental_impact:climate_change:production:landing_gear"
        ].value[0]
        + alpha_standard_datafile[
            "data:environmental_impact:climate_change:production:vertical_tail"
        ].value[0]
        + alpha_standard_datafile["data:environmental_impact:climate_change:production:wing"].value[
            0
        ]
    )

    print("Impact of airframe: ", airframe_impact)

    battery_impact = (
        alpha_standard_datafile[
            "data:environmental_impact:climate_change:production:battery_pack_1"
        ].value[0]
        + alpha_standard_datafile[
            "data:environmental_impact:climate_change:production:battery_pack_2"
        ].value[0]
    )

    print("Impact of battery: ", battery_impact)

    electricity_impact = (
        alpha_standard_datafile[
            "data:environmental_impact:climate_change:operation:electricity_for_mission"
        ].value[0]
        + alpha_standard_datafile[
            "data:environmental_impact:climate_change:production:assembly"
        ].value[0]
    )

    print("Impact of electricity: ", electricity_impact)

    impact_others = (
        alpha_standard_datafile["data:environmental_impact:climate_change:sum"].value[0]
        - airframe_impact
        - battery_impact
        - electricity_impact
    )

    print("Impact of others: ", impact_others)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_extract_results_from_pipistrel_lca_comparison_short():
    alpha_short_datafile = oad.DataFile(RESULTS_FOLDER_PATH / "pipistrel_alpha_short.xml")

    airframe_impact = (
        alpha_short_datafile[
            "data:environmental_impact:climate_change:production:flight_controls"
        ].value[0]
        + alpha_short_datafile[
            "data:environmental_impact:climate_change:production:fuselage"
        ].value[0]
        + alpha_short_datafile[
            "data:environmental_impact:climate_change:production:horizontal_tail"
        ].value[0]
        + alpha_short_datafile[
            "data:environmental_impact:climate_change:production:landing_gear"
        ].value[0]
        + alpha_short_datafile[
            "data:environmental_impact:climate_change:production:vertical_tail"
        ].value[0]
        + alpha_short_datafile["data:environmental_impact:climate_change:production:wing"].value[0]
    )

    print("Impact of airframe: ", airframe_impact)

    battery_impact = (
        alpha_short_datafile[
            "data:environmental_impact:climate_change:production:battery_pack_1"
        ].value[0]
        + alpha_short_datafile[
            "data:environmental_impact:climate_change:production:battery_pack_2"
        ].value[0]
    )

    print("Impact of battery: ", battery_impact)

    electricity_impact = (
        alpha_short_datafile[
            "data:environmental_impact:climate_change:operation:electricity_for_mission"
        ].value[0]
        + alpha_short_datafile[
            "data:environmental_impact:climate_change:production:assembly"
        ].value[0]
    )

    print("Impact of electricity: ", electricity_impact)

    impact_others = (
        alpha_short_datafile["data:environmental_impact:climate_change:sum"].value[0]
        - airframe_impact
        - battery_impact
        - electricity_impact
    )

    print("Impact of others: ", impact_others)


def test_kerosene_per_fu_tbm900():
    inputs_list = [
        "data:environmental_impact:flight_per_fu",
        "data:environmental_impact:aircraft_per_fu",
        "data:environmental_impact:line_test:mission_ratio",
        "data:environmental_impact:delivery:mission_ratio",
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_main_route",
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_2:fuel_consumed_main_route",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, DATA_FOLDER_PATH / "tbm900.xml")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAKerosenePerFU(
            tanks_name_list=["fuel_tank_1", "fuel_tank_2"],
            tanks_type_list=["fuel_tank", "fuel_tank"],
        ),
        ivc,
    )

    assert problem.get_val(
        "data:LCA:operation:he_power_train:kerosene:mass_per_fu"
    ) == pytest.approx(0.06860569, rel=1e-5)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:kerosene:mass_per_fu"
    ) == pytest.approx(9.39803907e-05, rel=1e-5)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:kerosene:mass_per_fu"
    ) == pytest.approx(5.63882344e-05, rel=1e-5)
    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_tbm900():
    ivc = get_indep_var_comp(
        list_inputs(
            LCA(
                power_train_file_path=DATA_FOLDER_PATH / "tbm900_propulsion.yml",
                component_level_breakdown=True,
                airframe_material="aluminium",
                delivery_method="flight",
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
            delivery_method="flight",
        ),
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "tbm900_lca.xml"
    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:sum"
    ) == pytest.approx(0.00018677255609807612, rel=1e-5)

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:turboshaft_1"
    ) == pytest.approx(6.3786e-5, rel=1e-3)

    assert problem.get_val(
        "data:environmental_impact:climate_change:operation:turboshaft_1"
    ) == pytest.approx(0.21645, rel=1e-3)

    assert problem.get_val(
        "data:environmental_impact:climate_change:operation:sum"
    ) == pytest.approx(0.27663, rel=1e-3)

    assert problem.get_val(
        "data:environmental_impact:climate_change:operation:turboshaft_1"
    ) + problem.get_val(
        "data:environmental_impact:climate_change:operation:kerosene_for_mission"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:climate_change:operation:sum"), rel=1e-3
    )

    assert problem.get_val(
        "data:environmental_impact:climate_change:manufacturing:sum"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:line_test:duration", units="h")
        / problem.get_val("data:mission:sizing:main_route:duration", units="h")
        * problem.get_val("data:environmental_impact:climate_change:operation:sum")
        * problem.get_val("data:environmental_impact:aircraft_per_fu")
        / problem.get_val("data:environmental_impact:flight_per_fu"),
        abs=1e-5,
    )

    assert problem.get_val(
        "data:environmental_impact:climate_change:manufacturing:sum"
    ) == pytest.approx(0.000109653, rel=1e-4)

    assert problem.get_val(
        "data:environmental_impact:climate_change:distribution:sum"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:climate_change:manufacturing:sum")
        / problem.get_val("data:environmental_impact:line_test:mission_ratio")
        * problem.get_val("data:environmental_impact:delivery:mission_ratio"),
        abs=1e-5,
    )

    assert problem.get_val(
        "data:environmental_impact:climate_change:distribution:sum"
    ) == pytest.approx(0.000198416, rel=1e-4)

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_tbm900_ef():
    ivc = get_indep_var_comp(
        list_inputs(
            LCA(
                power_train_file_path=DATA_FOLDER_PATH / "tbm900_propulsion.yml",
                component_level_breakdown=True,
                airframe_material="aluminium",
                delivery_method="flight",
                impact_assessment_method="EF v3.1",
                normalization=True,
                weighting=True,
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
            delivery_method="flight",
            impact_assessment_method="EF v3.1",
            normalization=True,
            weighting=True,
        ),
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "tbm900_lca_ef.xml"
    problem.write_outputs()

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:sum"
    ) == pytest.approx(0.000182843387198129, rel=1e-5)
    assert problem.get_val(
        "data:environmental_impact:climate_change_normalized:production:sum"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:climate_change:production:sum")
        / problem.get_val("data:environmental_impact:climate_change:normalization_factor"),
        rel=1e-5,
    )
    assert problem.get_val(
        "data:environmental_impact:climate_change_normalized:production:sum"
    ) == pytest.approx(2.4217667178560132e-08, rel=1e-5)
    assert problem.get_val(
        "data:environmental_impact:climate_change_weighted:sum"
    ) == pytest.approx(7.613406896780802e-06, rel=1e-5)
    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        1.746314961195058e-05, rel=1e-5
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_tbm900_ef_inputs_as_hours():
    ivc = get_indep_var_comp(
        list_inputs(
            LCA(
                power_train_file_path=DATA_FOLDER_PATH / "tbm900_propulsion.yml",
                component_level_breakdown=True,
                airframe_material="aluminium",
                delivery_method="flight",
                impact_assessment_method="EF v3.1",
                normalization=True,
                weighting=True,
                aircraft_lifespan_in_hours=True,
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
            delivery_method="flight",
            impact_assessment_method="EF v3.1",
            normalization=True,
            weighting=True,
            aircraft_lifespan_in_hours=True,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:environmental_impact:climate_change:production:sum"
    ) == pytest.approx(0.000182843387198129, rel=1e-5)


def test_gasoline_per_fu_sr22():
    inputs_list = [
        "data:environmental_impact:flight_per_fu",
        "data:environmental_impact:aircraft_per_fu",
        "data:environmental_impact:delivery:mission_ratio",
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_main_route",
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_2:fuel_consumed_main_route",
        "data:environmental_impact:line_test:mission_ratio",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, DATA_FOLDER_PATH / "sr22.xml")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCAGasolinePerFU(
            tanks_name_list=["fuel_tank_1", "fuel_tank_2"],
            tanks_type_list=["fuel_tank", "fuel_tank"],
        ),
        ivc,
    )

    assert problem.get_val(
        "data:LCA:operation:he_power_train:gasoline:mass_per_fu"
    ) == pytest.approx(0.02942765, rel=1e-5)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:gasoline:mass_per_fu"
    ) == pytest.approx(4.03118507e-05, rel=1e-5)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:gasoline:mass_per_fu"
    ) == pytest.approx(2.39111932e-05, rel=1e-5)

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_cirrus_sr22():
    ivc = get_indep_var_comp(
        list_inputs(
            LCA(
                power_train_file_path=DATA_FOLDER_PATH / "sr22_propulsion.yml",
                component_level_breakdown=True,
                airframe_material="composite",
                delivery_method="flight",
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
            delivery_method="flight",
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

    assert problem.get_val(
        "data:environmental_impact:climate_change:operation:ice_1"
    ) == pytest.approx(0.09122571820320151, rel=1e-3)

    assert problem.get_val(
        "data:environmental_impact:climate_change:operation:sum"
    ) == pytest.approx(0.12129755, rel=1e-3)

    assert problem.get_val(
        "data:environmental_impact:climate_change:manufacturing:sum"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:line_test:duration", units="h")
        / problem.get_val("data:mission:sizing:main_route:duration", units="h")
        * problem.get_val("data:environmental_impact:climate_change:operation:sum")
        * problem.get_val("data:environmental_impact:aircraft_per_fu")
        / problem.get_val("data:environmental_impact:flight_per_fu"),
        abs=1e-5,
    )

    assert problem.get_val(
        "data:environmental_impact:climate_change:manufacturing:sum"
    ) == pytest.approx(3.07602591e-05, rel=1e-4)

    assert problem.get_val(
        "data:environmental_impact:climate_change:distribution:sum"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:climate_change:manufacturing:sum")
        / problem.get_val("data:environmental_impact:line_test:mission_ratio")
        * problem.get_val("data:environmental_impact:delivery:mission_ratio"),
        abs=1e-5,
    )

    assert problem.get_val(
        "data:environmental_impact:climate_change:distribution:sum"
    ) == pytest.approx(9.85593183e-05, rel=1e-4)

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_kodiak_100_ef_and_hybrid():
    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion.yml",
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="flight",
        impact_assessment_method="EF v3.1",
        normalization=True,
        weighting=True,
        use_operational_mission=True,
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        DATA_FOLDER_PATH / "oad_process_outputs_op.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "kodiak_100_ef.xml"
    problem.write_outputs()

    # Second LCA

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "hybrid_propulsion.yml",
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="flight",
        impact_assessment_method="EF v3.1",
        normalization=True,
        weighting=True,
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        DATA_FOLDER_PATH / "oad_process_outputs_he.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "hybrid_kodiak_100_ef.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_kodiak_100_recipe_end_and_mid():
    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion.yml",
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="flight",
        normalization=True,
        weighting=True,
        use_operational_mission=True,
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        DATA_FOLDER_PATH / "oad_process_outputs_op.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "kodiak_100_recipe_end.xml"
    problem.write_outputs()

    # TODO: This test is disabled because I'm not sure about my equivalent midpoint weighting factor
    #  for ReCiPe
    # Takes time to reinitialize but I can't avoid it unfortunately
    # component = LCA(
    #     power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion.yml",
    #     component_level_breakdown=True,
    #     airframe_material="aluminium",
    #     delivery_method="flight",
    #     normalization=True,
    #     weighting=True,
    #     use_operational_mission=True,
    #     recipe_midpoint_weighting=True,
    # )
    #
    # # Run problem and check obtained value(s) is/(are) correct
    # problem = run_system(
    #     component,
    #     ivc,
    # )
    #
    # problem.output_file_path = RESULTS_FOLDER_PATH / "kodiak_100_recipe_mid.xml"
    # problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_hybrid_kodiak_100_recipe_mid_and_end():
    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "hybrid_propulsion.yml",
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="flight",
        normalization=True,
        weighting=True,
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        DATA_FOLDER_PATH / "oad_process_outputs_he.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "hybrid_kodiak_recipe_end.xml"
    problem.write_outputs()

    # TODO: This test is disabled because I'm not sure about my equivalent midpoint weighting factor
    #  for ReCiPe
    # Takes time to reinitialize but I can't avoid it unfortunately
    # component = LCA(
    #     power_train_file_path=DATA_FOLDER_PATH / "hybrid_propulsion.yml",
    #     component_level_breakdown=True,
    #     airframe_material="aluminium",
    #     delivery_method="flight",
    #     normalization=True,
    #     weighting=True,
    #     use_operational_mission=False,
    #     recipe_midpoint_weighting=True,
    # )
    #
    # # Run problem and check obtained value(s) is/(are) correct
    # problem = run_system(
    #     component,
    #     ivc,
    # )
    #
    # problem.output_file_path = RESULTS_FOLDER_PATH / "hybrid_kodiak_recipe_mid.xml"
    # problem.write_outputs()
