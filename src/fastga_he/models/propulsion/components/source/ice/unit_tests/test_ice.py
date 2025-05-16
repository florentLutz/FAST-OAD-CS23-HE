# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pytest

import openmdao.api as om

from stdatm import Atmosphere

from ..components.cstr_enforce import ConstraintsSeaLevelPowerEnforce
from ..components.cstr_ensure import ConstraintsSeaLevelPowerEnsure
from ..components.cstr_ice import ConstraintICEPowerRateMission

from ..components.sizing_displacement_volume import SizingICEDisplacementVolume
from ..components.sizing_ice_uninstalled_weight import SizingICEUninstalledWeight
from ..components.sizing_ice_weight import SizingICEWeight
from ..components.sizing_ice_dimensions_scaling import SizingICEDimensionsScaling
from ..components.sizing_ice_dimensions import SizingICEDimensions
from ..components.sizing_ice_nacelle_dimensions import SizingICENacelleDimensions
from ..components.sizing_ice_nacelle_wet_area import SizingICENacelleWetArea
from ..components.sizing_ice_drag import SizingICEDrag
from ..components.sizing_ice_cg_x import SizingICECGX
from ..components.sizing_ice_cg_y import SizingICECGY

from ..components.pre_lca_prod_weight_per_fu import PreLCAICEProdWeightPerFU
from ..components.pre_lca_use_emission_per_fu import PreLCAICEUseEmissionPerFU

from ..components.lcc_ice_cost import LCCICECost
from ..components.lcc_ice_operational_cost import LCCICEOperationalCost

from ..components.perf_torque import PerformancesTorque
from ..components.perf_equivalent_sl_power import PerformancesEquivalentSeaLevelPower
from ..components.perf_mean_effective_pressure import PerformancesMeanEffectivePressure
from ..components.perf_sfc import PerformancesSFC
from ..components.perf_fuel_consumption import PerformancesICEFuelConsumption
from ..components.perf_fuel_consumed import PerformancesICEFuelConsumed
from ..components.perf_maximum import PerformancesMaximum

from ..components.perf_inflight_co2_emissions import PerformancesICEInFlightCO2Emissions
from ..components.perf_inflight_co_emissions import PerformancesICEInFlightCOEmissions
from ..components.perf_inflight_nox_emissions import PerformancesICEInFlightNOxEmissions
from ..components.perf_inflight_sox_emissions import PerformancesICEInFlightSOxEmissions
from ..components.perf_inflight_h2o_emissions import PerformancesICEInFlightH2OEmissions
from ..components.perf_inflight_hc_emissions import PerformancesICEInFlightHCEmissions
from ..components.perf_inflight_lead_emissions import PerformancesICEInFlightLeadEmissions
from ..components.perf_inflight_emissions_sum import PerformancesICEInFlightEmissionsSum
from ..components.perf_inflight_emissions import PerformancesICEInFlightEmissions

from ..components.sizing_ice import SizingICE
from ..components.perf_ice import PerformancesICE

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_ice.xml"
NB_POINTS_TEST = 10


def test_constraint_power_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsSeaLevelPowerEnforce(ice_id="ice_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsSeaLevelPowerEnforce(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:power_rating_SL", units="kW"
    ) == pytest.approx(245.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsSeaLevelPowerEnsure(ice_id="ice_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsSeaLevelPowerEnsure(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:ICE:ice_1:power_rating_SL", units="kW"
    ) == pytest.approx(-5.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintICEPowerRateMission(ice_id="ice_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintICEPowerRateMission(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:shaft_power_rating", units="kW"
    ) == pytest.approx(250.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_displacement_volume():
    ivc = get_indep_var_comp(
        list_inputs(SizingICEDisplacementVolume(ice_id="ice_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingICEDisplacementVolume(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:displacement_volume", units="m**3"
    ) == pytest.approx(0.0107, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_uninstalled_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingICEUninstalledWeight(ice_id="ice_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingICEUninstalledWeight(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:uninstalled_mass", units="kg"
    ) == pytest.approx(258.013, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_installed_weight():
    ivc = get_indep_var_comp(list_inputs(SizingICEWeight(ice_id="ice_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingICEWeight(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:mass", units="kg"
    ) == pytest.approx(309.61, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_installed_dimensions_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingICEDimensionsScaling(ice_id="ice_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingICEDimensionsScaling(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:scaling:length"
    ) == pytest.approx(1.235, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:scaling:width"
    ) == pytest.approx(1.235, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:scaling:height"
    ) == pytest.approx(1.235, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_ice_dimensions():
    ivc = get_indep_var_comp(list_inputs(SizingICEDimensions(ice_id="ice_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingICEDimensions(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:engine:length", units="m"
    ) == pytest.approx(1.03, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:engine:width", units="m"
    ) == pytest.approx(1.05, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:engine:height", units="m"
    ) == pytest.approx(0.704, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_nacelle_dimensions():
    ivc = get_indep_var_comp(
        list_inputs(SizingICENacelleDimensions(ice_id="ice_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingICENacelleDimensions(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:nacelle:length", units="m"
    ) == pytest.approx(2.06, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:nacelle:width", units="m"
    ) == pytest.approx(1.16, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:nacelle:height", units="m"
    ) == pytest.approx(0.774, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_nacelle_wet_area():
    ivc = get_indep_var_comp(
        list_inputs(SizingICENacelleWetArea(ice_id="ice_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingICENacelleWetArea(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:nacelle:wet_area", units="m**2"
    ) == pytest.approx(7.92688, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_nacelle_drag():
    expected_drag_ls = [5.38, 0.0, 0.0]
    expected_drag_cruise = [5.32, 0.0, 0.0]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_drag_ls, expected_drag_cruise
    ):
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingICEDrag(ice_id="ice_1", position=option, low_speed_aero=ls_option)
                ),
                __file__,
                XML_FILE,
            )
            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingICEDrag(ice_id="ice_1", position=option, low_speed_aero=ls_option), ivc
            )

            if ls_option:
                assert problem.get_val(
                    "data:propulsion:he_power_train:ICE:ice_1:low_speed:CD0",
                ) * 1e3 == pytest.approx(ls_drag, rel=1e-2)
            else:
                assert problem.get_val(
                    "data:propulsion:he_power_train:ICE:ice_1:cruise:CD0",
                ) * 1e3 == pytest.approx(cruise_drag, rel=1e-2)

            # Slight error on reynolds is due to step
            problem.check_partials(compact_print=True)


def test_motor_cg_x():
    expected_cg = [3.03, 1.03, 3.00]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingICECGX(ice_id="ice_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingICECGX(ice_id="ice_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:ICE:ice_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_cg_y():
    expected_cg = [2.0, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingICECGY(ice_id="ice_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingICECGY(ice_id="ice_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:ICE:ice_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_ice_sizing():
    ivc = get_indep_var_comp(list_inputs(SizingICE(ice_id="ice_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingICE(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:mass", units="kg"
    ) == pytest.approx(302.59, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:CG:x", units="m"
    ) == pytest.approx(3.03, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:CG:y", units="m"
    ) == pytest.approx(2.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:low_speed:CD0",
    ) * 1e3 == pytest.approx(5.32, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(150.0, 250.0, NB_POINTS_TEST),
        units="kW",
    )
    ivc.add_output(
        "rpm",
        val=np.linspace(2500.0, 2700.0, NB_POINTS_TEST),
        units="min**-1",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorque(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_out", units="N*m") == pytest.approx(
        np.array([573.0, 610.0, 646.0, 682.0, 717.0, 752.0, 786.0, 819.0, 852.0, 884.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_equivalent_power():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(150.0, 250.0, NB_POINTS_TEST),
        units="kW",
    )
    altitude = np.linspace(8000.0, 0.0, NB_POINTS_TEST)
    ivc.add_output(
        "density",
        val=Atmosphere(altitude, altitude_in_feet=True).density,
        units="kg/m**3",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesEquivalentSeaLevelPower(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("equivalent_SL_power", units="kW") == pytest.approx(
        np.array([198.0, 206.0, 213.0, 220.0, 226.0, 232.0, 237.0, 242.0, 246.0, 250.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_mean_effective_pressure():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesMeanEffectivePressure(ice_id="ice_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "torque_out",
        val=np.array([573.0, 610.0, 646.0, 682.0, 717.0, 752.0, 786.0, 819.0, 852.0, 884.0]),
        units="N*m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMeanEffectivePressure(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("mean_effective_pressure", units="bar") == pytest.approx(
        np.array([13.5, 14.3, 15.2, 16.0, 16.8, 17.7, 18.5, 19.2, 20.0, 20.8]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_sfc():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "mean_effective_pressure",
        val=np.array([13.5, 14.3, 15.2, 16.0, 16.8, 17.7, 18.5, 19.2, 20.0, 20.6]),
        units="bar",
    )
    ivc.add_output(
        "rpm",
        val=np.linspace(2500.0, 2699.0, NB_POINTS_TEST),
        units="min**-1",
    )
    # If we put 2700.0 the check_partials will clip as intended but cause an unrealistically high
    # value for partial check

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesSFC(number_of_points=NB_POINTS_TEST, ice_id="ice_1"), ivc)

    assert problem.get_val("specific_fuel_consumption", units="g/kW/h") == pytest.approx(
        np.array([246.0, 246.0, 247.0, 249.0, 252.0, 257.0, 262.0, 267.0, 274.0, 280.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumption():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "specific_fuel_consumption",
        val=np.array([246.0, 246.0, 247.0, 249.0, 252.0, 257.0, 262.0, 267.0, 274.0, 280.0]),
        units="g/kW/h",
    )
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(150.0, 250.0, NB_POINTS_TEST),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesICEFuelConsumption(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("fuel_consumption", units="kg/h") == pytest.approx(
        np.array([36.9, 39.6, 42.5, 45.6, 49.0, 52.8, 56.8, 60.8, 65.5, 70.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumed():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumption",
        val=np.array([36.9, 39.6, 42.5, 45.6, 49.0, 52.8, 56.8, 60.8, 65.5, 70.0]),
        units="kg/h",
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesICEFuelConsumed(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "equivalent_SL_power",
        val=np.array([198.0, 206.0, 213.0, 220.0, 226.0, 232.0, 237.0, 242.0, 246.0, 250.0]),
        units="kW",
    )
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(150.0, 250.0, NB_POINTS_TEST),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesMaximum(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:power_max_SL", units="W"
    ) == pytest.approx(250e3, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:shaft_power_max", units="W"
    ) == pytest.approx(250e3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances_ice():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesICE(ice_id="ice_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(150.0, 250.0, NB_POINTS_TEST),
        units="kW",
    )
    ivc.add_output(
        "rpm",
        val=np.linspace(2500.0, 2699.0, NB_POINTS_TEST),
        units="min**-1",
    )
    altitude = np.linspace(8000.0, 0.0, NB_POINTS_TEST)
    ivc.add_output(
        "density",
        val=Atmosphere(altitude, altitude_in_feet=True).density,
        units="kg/m**3",
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesICE(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        rel=1e-2,
    )
    assert problem.get_val("non_consumable_energy_t", units="kW*h") == pytest.approx(
        np.zeros(NB_POINTS_TEST),
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:power_max_SL", units="W"
    ) == pytest.approx(250e3, rel=1e-2)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_in_flight_co2_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesICEInFlightCO2Emissions(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("CO2_emissions", units="g") == pytest.approx(
        np.array(
            [
                15872.0,
                17050.0,
                18290.0,
                19623.0,
                21111.0,
                22723.0,
                24459.0,
                26164.0,
                28210.0,
                30132.0,
            ]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_co_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesICEInFlightCOEmissions(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("CO_emissions", units="g") == pytest.approx(
        np.array(
            [4085.76, 4389.0, 4708.2, 5051.34, 5434.38, 5849.34, 6296.22, 6735.12, 7261.8, 7756.56]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_nox_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesICEInFlightNOxEmissions(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("NOx_emissions", units="g") == pytest.approx(
        np.array(
            [16.0768, 17.27, 18.526, 19.8762, 21.3834, 23.0162, 24.7746, 26.5016, 28.574, 30.5208]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_sox_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesICEInFlightSOxEmissions(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("SOx_emissions", units="g") == pytest.approx(
        np.array([2.1504, 2.31, 2.478, 2.6586, 2.8602, 3.0786, 3.3138, 3.5448, 3.822, 4.0824]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_h2o_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesICEInFlightH2OEmissions(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("H2O_emissions", units="g") == pytest.approx(
        np.array(
            [
                6333.44,
                6803.5,
                7298.3,
                7830.21,
                8423.97,
                9067.21,
                9759.93,
                10440.28,
                11256.7,
                12023.64,
            ]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_hc_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesICEInFlightHCEmissions(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("HC_emissions", units="g") == pytest.approx(
        np.array(
            [
                96.59904,
                103.7685,
                111.3153,
                119.42811,
                128.48427,
                138.29511,
                148.86063,
                159.23748,
                171.6897,
                183.38724,
            ]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_lead_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesICEInFlightLeadEmissions(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("lead_emissions", units="g") == pytest.approx(
        np.array(
            [4.06528, 4.367, 4.6846, 5.02602, 5.40714, 5.82002, 6.26466, 6.70136, 7.2254, 7.71768]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_emissions_sum():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "CO2_emissions",
        units="g",
        val=np.array(
            [
                15872.0,
                17050.0,
                18290.0,
                19623.0,
                21111.0,
                22723.0,
                24459.0,
                26164.0,
                28210.0,
                30132.0,
            ]
        ),
    )
    ivc.add_output(
        "CO_emissions",
        units="g",
        val=np.array(
            [4085.76, 4389.0, 4708.2, 5051.34, 5434.38, 5849.34, 6296.22, 6735.12, 7261.8, 7756.56]
        ),
    )
    ivc.add_output(
        "NOx_emissions",
        units="g",
        val=np.array(
            [16.0768, 17.27, 18.526, 19.8762, 21.3834, 23.0162, 24.7746, 26.5016, 28.574, 30.5208]
        ),
    )
    ivc.add_output(
        "SOx_emissions",
        units="g",
        val=np.array([2.1504, 2.31, 2.478, 2.6586, 2.8602, 3.0786, 3.3138, 3.5448, 3.822, 4.0824]),
    )
    ivc.add_output(
        "H2O_emissions",
        units="g",
        val=np.array(
            [
                6333.44,
                6803.5,
                7298.3,
                7830.21,
                8423.97,
                9067.21,
                9759.93,
                10440.28,
                11256.7,
                12023.64,
            ]
        ),
    )
    ivc.add_output(
        "HC_emissions",
        units="g",
        val=np.array(
            [
                96.59904,
                103.7685,
                111.3153,
                119.42811,
                128.48427,
                138.29511,
                148.86063,
                159.23748,
                171.6897,
                183.38724,
            ]
        ),
    )
    ivc.add_output(
        "lead_emissions",
        units="g",
        val=np.array(
            [4.06528, 4.367, 4.6846, 5.02602, 5.40714, 5.82002, 6.26466, 6.70136, 7.2254, 7.71768]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesICEInFlightEmissionsSum(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:CO2", units="kg"
    ) == pytest.approx(223.634, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:CO", units="kg"
    ) == pytest.approx(57.56, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:NOx", units="g"
    ) == pytest.approx(226.5196, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:SOx", units="g"
    ) == pytest.approx(30.2988, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:H2O", units="kg"
    ) == pytest.approx(89.237, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:HC", units="g"
    ) == pytest.approx(1361.06, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:lead", units="g"
    ) == pytest.approx(57.27916, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_in_flight_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesICEInFlightEmissions(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:CO2", units="kg"
    ) == pytest.approx(223.634, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:CO", units="kg"
    ) == pytest.approx(57.56, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:NOx", units="g"
    ) == pytest.approx(226.5196, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:SOx", units="g"
    ) == pytest.approx(30.2988, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:H2O", units="kg"
    ) == pytest.approx(89.237, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:HC", units="g"
    ) == pytest.approx(1361.06, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:ICE:ice_1:mass",
        "data:environmental_impact:aircraft_per_fu",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCAICEProdWeightPerFU(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:mass_per_fu", units="kg"
    ) == pytest.approx(0.00036122, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_emissions_per_fu():
    inputs_list = [
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:CO2",
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:CO",
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:NOx",
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:SOx",
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:HC",
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:H2O",
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:lead",
        "data:environmental_impact:flight_per_fu",
        "data:environmental_impact:aircraft_per_fu",
        "data:environmental_impact:line_test:mission_ratio",
        "data:environmental_impact:delivery:mission_ratio",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCAICEUseEmissionPerFU(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:LCA:operation:he_power_train:ICE:ice_1:CO2_per_fu", units="kg"
    ) == pytest.approx(0.72375211, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:ICE:ice_1:CO_per_fu", units="kg"
    ) == pytest.approx(0.1863078, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:ICE:ice_1:NOx_per_fu", units="kg"
    ) == pytest.approx(0.00073309, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:ICE:ice_1:SOx_per_fu", units="kg"
    ) == pytest.approx(9.80567381e-05, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:ICE:ice_1:H2O_per_fu", units="kg"
    ) == pytest.approx(0.28880044, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:ICE:ice_1:HC_per_fu", units="kg"
    ) == pytest.approx(0.00440485, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:ICE:ice_1:lead_per_fu",
        units="kg",
    ) == pytest.approx(5.727916e-05, rel=1e-3)

    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:ICE:ice_1:CO2_per_fu", units="kg"
    ) == pytest.approx(0.005428125, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:ICE:ice_1:CO_per_fu", units="kg"
    ) == pytest.approx(0.0013973085, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:ICE:ice_1:NOx_per_fu", units="kg"
    ) == pytest.approx(5.498174999999999e-06, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:ICE:ice_1:SOx_per_fu", units="kg"
    ) == pytest.approx(7.3542553575e-07, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:ICE:ice_1:H2O_per_fu", units="kg"
    ) == pytest.approx(0.0021660033, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:ICE:ice_1:HC_per_fu", units="kg"
    ) == pytest.approx(3.3036375e-05, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:ICE:ice_1:lead_per_fu", units="kg"
    ) == pytest.approx(4.295937e-07, rel=1e-3)

    assert problem.get_val(
        "data:LCA:distribution:he_power_train:ICE:ice_1:CO2_per_fu", units="kg"
    ) == pytest.approx(0.00289501, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:ICE:ice_1:CO_per_fu", units="kg"
    ) == pytest.approx(0.00074523, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:ICE:ice_1:NOx_per_fu", units="kg"
    ) == pytest.approx(2.9323634e-06, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:ICE:ice_1:SOx_per_fu", units="kg"
    ) == pytest.approx(3.92226952e-07, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:ICE:ice_1:H2O_per_fu", units="kg"
    ) == pytest.approx(0.0011552, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:ICE:ice_1:HC_per_fu", units="kg"
    ) == pytest.approx(1.7619395e-05, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:ICE:ice_1:lead_per_fu", units="kg"
    ) == pytest.approx(2.2911664e-07, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cost():
    ivc = om.IndepVarComp()
    ivc.add_output("data:cost:cpi_2012", val=1.4)
    ivc.add_output(
        "data:propulsion:he_power_train:ICE:ice_1:power_max_SL",
        val=250.0,
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCICECost(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:purchase_cost", units="USD"
    ) == pytest.approx(
        81668.231,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_operational_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:ICE:ice_1:displacement_volume",
        val=0.0107,
        units="m**3",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCICEOperationalCost(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:operational_cost", units="USD/yr"
    ) == pytest.approx(
        9887.5,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:ICE:ice_1:displacement_volume",
        val=0.0007,
        units="m**3",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCICEOperationalCost(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:operational_cost", units="USD/yr"
    ) == pytest.approx(
        0.00106987,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)
