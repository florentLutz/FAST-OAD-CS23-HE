# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pathlib

import numpy as np
import pytest

import openmdao.api as om

from stdatm import Atmosphere

from ..components.cstr_enforce import ConstraintsSeaLevelPowerEnforce
from ..components.cstr_ensure import ConstraintsSeaLevelPowerEnsure
from ..components.cstr_high_rpm_ice import ConstraintHighRPMICEPowerRateMission

from ..components.sizing_displacement_volume import SizingHighRPMICEDisplacementVolume
from ..components.sizing_sfc_min_mep import SizingHighRPMICESFCMinMEP
from ..components.sizing_sfc_max_mep import SizingHighRPMICESFCMaxMEP
from ..components.sizing_sfc_k_coefficient import SizingHighRPMICESFCKCoefficient

from ..components.perf_mean_effective_pressure import PerformancesMeanEffectivePressure
from ..components.perf_sfc import PerformancesSFC

from ..components.perf_inflight_co2_emissions import PerformancesHighRPMICEInFlightCO2Emissions
from ..components.perf_inflight_co_emissions import PerformancesHighRPMICEInFlightCOEmissions
from ..components.perf_inflight_nox_emissions import PerformancesHighRPMICEInFlightNOxEmissions
from ..components.perf_inflight_sox_emissions import PerformancesHighRPMICEInFlightSOxEmissions
from ..components.perf_inflight_h2o_emissions import PerformancesHighRPMICEInFlightH2OEmissions
from ..components.perf_inflight_hc_emissions import PerformancesHighRPMICEInFlightHCEmissions
from ..components.perf_inflight_lead_emissions import PerformancesHighRPMICEInFlightLeadEmissions
from ..components.perf_inflight_emissions_sum import PerformancesHighRPMICEInFlightEmissionsSum
from ..components.perf_inflight_emissions import PerformancesHighRPMICEInFlightEmissions
from ..components.perf_maximum import PerformancesMaximum

from ...ice.components.perf_fuel_consumption import PerformancesICEFuelConsumption
from ...ice.components.perf_fuel_consumed import PerformancesICEFuelConsumed
from ...ice.components.perf_equivalent_sl_power import PerformancesEquivalentSeaLevelPower

from ..components.perf_high_rpm_ice import PerformancesHighRPMICE

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_high_rpm_ice.xml"
NB_POINTS_TEST = 10


def test_constraint_power_enforce():
    ivc = get_indep_var_comp(
        ["data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_max_SL"], __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsSeaLevelPowerEnforce(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL", units="kW"
    ) == pytest.approx(68.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_ensure():
    ivc = get_indep_var_comp(
        [
            "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_max_SL",
            "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL",
        ],
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsSeaLevelPowerEnsure(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL", units="kW"
    ) == pytest.approx(-1.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():
    ivc = get_indep_var_comp(
        ["data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL"], __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintHighRPMICEPowerRateMission(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:shaft_power_rating", units="kW"
    ) == pytest.approx(69.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_displacement_volume():
    inputs_list = [
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL",
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:pme_max",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHighRPMICEDisplacementVolume(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:displacement_volume", units="cm**3"
    ) == pytest.approx(1352.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sfc_min_mep():
    ivc = get_indep_var_comp(
        ["data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL"], __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHighRPMICESFCMinMEP(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:sfc_coefficient:min_mep", units="g/kW/h"
    ) == pytest.approx(620.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sfc_max_mep():
    ivc = get_indep_var_comp(
        ["data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL"], __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHighRPMICESFCMaxMEP(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:sfc_coefficient:max_mep", units="g/kW/h"
    ) == pytest.approx(300.6, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sfc_k_coefficient():
    ivc = get_indep_var_comp(
        [
            "data:propulsion:he_power_train:high_rpm_ICE:ice_1:sfc_coefficient:max_mep",
            "data:propulsion:he_power_train:high_rpm_ICE:ice_1:sfc_coefficient:min_mep",
        ],
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHighRPMICESFCKCoefficient(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:sfc_coefficient:k_coefficient"
    ) == pytest.approx(0.443, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_equivalent_power():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(5.0, 50.0, NB_POINTS_TEST),
        units="kW",
    )
    altitude = np.linspace(4000.0, 0.0, NB_POINTS_TEST)
    ivc.add_output(
        "density",
        val=Atmosphere(altitude, altitude_in_feet=True).density,
        units="kg/m**3",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesEquivalentSeaLevelPower(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("equivalent_SL_power", units="kW") == pytest.approx(
        np.array([5.72, 11.2, 16.6, 21.8, 26.9, 31.8, 36.5, 41.2, 45.6, 50.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_mean_effective_pressure():
    ivc = get_indep_var_comp(
        ["data:propulsion:he_power_train:high_rpm_ICE:ice_1:displacement_volume"],
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "torque_out",
        val=np.array([30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]),
        units="N*m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMeanEffectivePressure(high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("mean_effective_pressure", units="bar") == pytest.approx(
        np.array([5.57, 7.43, 9.29, 11.1, 13.0, 14.8, 16.7, 18.5, 20.4, 22.3]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_sfc():
    ivc = get_indep_var_comp(
        [
            "data:propulsion:he_power_train:high_rpm_ICE:ice_1:sfc_coefficient:k_coefficient",
            "data:propulsion:he_power_train:high_rpm_ICE:ice_1:sfc_coefficient:max_mep",
        ],
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "mean_effective_pressure",
        val=np.array([4.5, 5.57, 7.43, 9.29, 11.1, 13.0, 14.8, 16.7, 18.5, 20.4]),
        units="bar",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSFC(high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("specific_fuel_consumption", units="g/kW/h") == pytest.approx(
        np.array([617.0, 546.0, 408.0, 347.0, 321.0, 309.0, 304.0, 302.0, 301.0, 300.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumption():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "specific_fuel_consumption",
        val=np.array([617.0, 546.0, 408.0, 347.0, 321.0, 309.0, 304.0, 302.0, 301.0, 300.0]),
        units="g/kW/h",
    )
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(5.0, 50.0, NB_POINTS_TEST),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesICEFuelConsumption(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("fuel_consumption", units="kg/h") == pytest.approx(
        np.array([3.085, 5.46, 6.12, 6.94, 8.025, 9.27, 10.64, 12.08, 13.545, 15.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumed():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumption",
        val=np.array([3.085, 5.46, 6.12, 6.94, 8.025, 9.27, 10.64, 12.08, 13.545, 15.0]),
        units="kg/h",
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesICEFuelConsumed(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([0.428, 0.758, 0.85, 0.96, 1.11, 1.28, 1.47, 1.67, 1.88, 2.08]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_co2_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([0.428, 0.758, 0.85, 0.96, 1.11, 1.28, 1.47, 1.67, 1.88, 2.08]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICEInFlightCO2Emissions(
            high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("CO2_emissions", units="g") == pytest.approx(
        np.array([1326.8, 2349.8, 2635.0, 2976.0, 3441.0, 3968.0, 4557.0, 5177.0, 5828.0, 6448.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_co_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([0.428, 0.758, 0.85, 0.96, 1.11, 1.28, 1.47, 1.67, 1.88, 2.08]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICEInFlightCOEmissions(
            high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("CO_emissions", units="g") == pytest.approx(
        np.array(
            [341.544, 604.884, 678.3, 766.08, 885.78, 1021.44, 1173.06, 1332.66, 1500.24, 1659.84]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_nox_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([0.428, 0.758, 0.85, 0.96, 1.11, 1.28, 1.47, 1.67, 1.88, 2.08]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICEInFlightNOxEmissions(
            high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("NOx_emissions", units="g") == pytest.approx(
        np.array([1.34392, 2.38012, 2.669, 3.0144, 3.4854, 4.0192, 4.6158, 5.2438, 5.9032, 6.5312]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_sox_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([0.428, 0.758, 0.85, 0.96, 1.11, 1.28, 1.47, 1.67, 1.88, 2.08]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICEInFlightSOxEmissions(
            high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("SOx_emissions", units="g") == pytest.approx(
        np.array([0.17976, 0.31836, 0.357, 0.4032, 0.4662, 0.5376, 0.6174, 0.7014, 0.7896, 0.8736]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_h2o_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([0.428, 0.758, 0.85, 0.96, 1.11, 1.28, 1.47, 1.67, 1.88, 2.08]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICEInFlightH2OEmissions(
            high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("H2O_emissions", units="g") == pytest.approx(
        np.array(
            [
                529.436,
                937.646,
                1051.45,
                1187.52,
                1373.07,
                1583.36,
                1818.39,
                2065.79,
                2325.56,
                2572.96,
            ]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_hc_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([0.428, 0.758, 0.85, 0.96, 1.11, 1.28, 1.47, 1.67, 1.88, 2.08]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICEInFlightHCEmissions(
            high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("HC_emissions", units="g") == pytest.approx(
        np.array(
            [
                8.075076,
                14.301186,
                16.03695,
                18.11232,
                20.94237,
                24.14976,
                27.73449,
                31.50789,
                35.46996,
                39.24336,
            ]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_lead_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([0.428, 0.758, 0.85, 0.96, 1.11, 1.28, 1.47, 1.67, 1.88, 2.08]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICEInFlightLeadEmissions(
            high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("lead_emissions", units="g") == pytest.approx(
        np.array(
            [
                0.339832,
                0.601852,
                0.6749,
                0.76224,
                0.88134,
                1.01632,
                1.16718,
                1.32598,
                1.49272,
                1.65152,
            ]
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
            [1326.8, 2349.8, 2635.0, 2976.0, 3441.0, 3968.0, 4557.0, 5177.0, 5828.0, 6448.0]
        ),
    )
    ivc.add_output(
        "CO_emissions",
        units="g",
        val=np.array(
            [341.544, 604.884, 678.3, 766.08, 885.78, 1021.44, 1173.06, 1332.66, 1500.24, 1659.84]
        ),
    )
    ivc.add_output(
        "NOx_emissions",
        units="g",
        val=np.array(
            [1.34392, 2.38012, 2.669, 3.0144, 3.4854, 4.0192, 4.6158, 5.2438, 5.9032, 6.5312]
        ),
    )
    ivc.add_output(
        "SOx_emissions",
        units="g",
        val=np.array(
            [0.17976, 0.31836, 0.357, 0.4032, 0.4662, 0.5376, 0.6174, 0.7014, 0.7896, 0.8736]
        ),
    )
    ivc.add_output(
        "H2O_emissions",
        units="g",
        val=np.array(
            [
                529.436,
                937.646,
                1051.45,
                1187.52,
                1373.07,
                1583.36,
                1818.39,
                2065.79,
                2325.56,
                2572.96,
            ]
        ),
    )
    ivc.add_output(
        "HC_emissions",
        units="g",
        val=np.array(
            [
                8.075076,
                14.301186,
                16.03695,
                18.11232,
                20.94237,
                24.14976,
                27.73449,
                31.50789,
                35.46996,
                39.24336,
            ]
        ),
    )
    ivc.add_output(
        "lead_emissions",
        units="g",
        val=np.array(
            [
                0.339832,
                0.601852,
                0.6749,
                0.76224,
                0.88134,
                1.01632,
                1.16718,
                1.32598,
                1.49272,
                1.65152,
            ]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICEInFlightEmissionsSum(
            high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO2",
        units="kg",
    ) == pytest.approx(38.70, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO",
        units="kg",
    ) == pytest.approx(9.96, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:NOx",
        units="g",
    ) == pytest.approx(39.20, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:SOx",
        units="g",
    ) == pytest.approx(5.24, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:H2O",
        units="kg",
    ) == pytest.approx(15.44, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:HC", units="g"
    ) == pytest.approx(235.57, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:lead",
        units="g",
    ) == pytest.approx(9.91, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_in_flight_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([0.428, 0.758, 0.85, 0.96, 1.11, 1.28, 1.47, 1.67, 1.88, 2.08]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICEInFlightEmissions(
            high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO2",
        units="kg",
    ) == pytest.approx(38.70, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO",
        units="kg",
    ) == pytest.approx(9.96, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:NOx",
        units="g",
    ) == pytest.approx(39.20, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:SOx",
        units="g",
    ) == pytest.approx(5.24, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:H2O",
        units="kg",
    ) == pytest.approx(15.44, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:HC", units="g"
    ) == pytest.approx(235.57, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "equivalent_SL_power",
        val=np.array([5.72, 11.2, 16.6, 21.8, 26.9, 31.8, 36.5, 41.2, 45.6, 50.0]),
        units="kW",
    )
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(5.0, 50.0, NB_POINTS_TEST),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_max_SL", units="W"
    ) == pytest.approx(50e3, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:shaft_power_max", units="W"
    ) == pytest.approx(50e3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances_ice():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesHighRPMICE(high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(5.0, 50.0, NB_POINTS_TEST),
        units="kW",
    )
    ivc.add_output(
        "rpm",
        val=np.linspace(2500.0, 5000.0, NB_POINTS_TEST),
        units="min**-1",
    )
    altitude = np.linspace(4000.0, 0.0, NB_POINTS_TEST)
    ivc.add_output(
        "density",
        val=Atmosphere(altitude, altitude_in_feet=True).density,
        units="kg/m**3",
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHighRPMICE(high_rpm_ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([0.42891051, 0.65531621, 0.75367227, 0.90704623, 1.08732801,
       1.28056926, 1.48034806, 1.68361652, 1.88885209, 2.09525266]),
        rel=1e-2,
    )
    assert problem.get_val("non_consumable_energy_t", units="kW*h") == pytest.approx(
        np.zeros(NB_POINTS_TEST),
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_max_SL", units="W"
    ) == pytest.approx(50e3, rel=1e-2)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pathlib.Path(__file__).parent / "n2.html")
