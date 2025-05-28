# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import pytest
import numpy as np
import openmdao.api as om

from ..components.cstr_enforce import ConstraintsRatedPowerEnforce
from ..components.cstr_ensure import ConstraintsRatedPowerEnsure
from ..components.cstr_turboshaft import ConstraintTurboshaftPowerRateMission

from ..components.sizing_turboshaft_uninstalled_weight import SizingTurboshaftUninstalledWeight
from ..components.sizing_turboshaft_weight import SizingTurboshaftWeight
from ..components.sizing_turboshaft_dimensions import SizingTurboshaftDimensions
from ..components.sizing_turboshaft_nacelle_dimensions import SizingTurboshaftNacelleDimensions
from ..components.sizing_turboshaft_nacelle_wet_area import SizingTurboshaftNacelleWetArea
from ..components.sizing_turboshaft_drag import SizingTurboshaftDrag
from ..components.sizing_turboshaft_cg_x import SizingTurboshaftCGX
from ..components.sizing_turboshaft_cg_y import SizingTurboshaftCGY

from ..components.pre_lca_prod_weight_per_fu import PreLCATurboshaftProdWeightPerFU
from ..components.pre_lca_use_emission_per_fu import PreLCATurboshaftUseEmissionPerFU

from ..components.lcc_turboshaft_cost import LCCTurboshaftCost
from ..components.lcc_turboshaft_operational_cost import LCCTurboshaftOperationalCost

from ..components.perf_density_ratio import PerformancesDensityRatio
from ..components.perf_mach import PerformancesMach
from ..components.perf_required_power import PerformancesRequiredPower
from ..components.perf_power_for_power_rate import PerformancesPowerForPowerRate
from ..components.perf_max_power_opr_limit import PerformancesMaxPowerOPRLimit
from ..components.perf_max_power_itt_limit import PerformancesMaxPowerITTLimit
from ..components.perf_equivalent_rated_power_itt_limit import (
    PerformancesEquivalentRatedPowerITTLimit,
)
from ..components.perf_equivalent_rated_power_opr_limit import (
    PerformancesEquivalentRatedPowerOPRLimit,
)
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_fuel_consumption import PerformancesTurboshaftFuelConsumption
from ..components.perf_fuel_consumed import PerformancesTurboshaftFuelConsumed
from ..components.perf_sfc import PerformancesSFC

from ..components.perf_inflight_co2_emissions import PerformancesTurboshaftInFlightCO2Emissions
from ..components.perf_inflight_h2o_emissions import PerformancesTurboshaftInFlightH2OEmissions
from ..components.perf_inflight_sox_emissions import PerformancesTurboshaftInFlightSOxEmissions
from ..components.perf_inflight_co_emissions import PerformancesTurboshaftInFlightCOEmissions
from ..components.perf_inflight_hc_emissions import PerformancesTurboshaftInFlightHCEmissions
from ..components.perf_inflight_nox_emissions import PerformancesTurboshaftInFlightNOxEmissions
from ..components.perf_inflight_emissions_sum import PerformancesTurboshaftInFlightEmissionsSum
from ..components.perf_inflight_emissions import PerformancesTurboshaftInFlightEmissions

from ..components.slipstream_density_ratio import SlipstreamDensityRatio
from ..components.slipstream_mach import SlipstreamMach
from ..components.slipstream_required_power import SlipstreamRequiredPower
from ..components.slipstream_exhaust_velocity import SlipstreamExhaustVelocity
from ..components.slipstream_exhaust_mass_flow import SlipstreamExhaustMassFlow
from ..components.slipstream_exhaust_thrust import SlipstreamExhaustThrust
from ..components.slipstream_delta_cd import SlipstreamTurboshaftDeltaCd

from ..components.sizing_turboshaft import SizingTurboshaft
from ..components.perf_turboshaft import PerformancesTurboshaft
from ..components.slipstream_turboshaft import SlipstreamTurboshaft

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_turboshaft.xml"
NB_POINTS_TEST = 10


def test_fuel_consumption_pw206b():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating", units="kW", val=308
    )
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:T41t",
        units="degK",
        val=1400.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:OPR", val=8.0
    )
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:power_ratio", val=1.56
    )
    ivc.add_output("density_ratio", val=1.0)
    ivc.add_output("mach", val=0.01)
    ivc.add_output("power_required", val=308, units="kW")

    problem = run_system(
        PerformancesTurboshaftFuelConsumption(turboshaft_id="turboshaft_1", number_of_points=1),
        ivc,
    )

    sfc = (
        problem.get_val("fuel_consumption", units="lb/h")[0]
        / problem.get_val("power_required", units="hp")[0]
    )
    print("k_sfc:", 0.548 / sfc)
    # Should be 0.548


def test_constraint_power_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRatedPowerEnforce(turboshaft_id="turboshaft_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRatedPowerEnforce(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating", units="kW"
    ) == pytest.approx(625, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRatedPowerEnsure(turboshaft_id="turboshaft_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRatedPowerEnsure(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating", units="kW"
    ) == pytest.approx(-9.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintTurboshaftPowerRateMission(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintTurboshaftPowerRateMission(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:shaft_power_rating", units="kW"
    ) == pytest.approx(634.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_uninstalled_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftUninstalledWeight(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftUninstalledWeight(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:uninstalled_mass", units="kg"
    ) == pytest.approx(192.9758, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_installed_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftWeight(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftWeight(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:mass", units="kg"
    ) == pytest.approx(231.57096, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_turboshaft_dimensions():
    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftDimensions(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftDimensions(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:engine:height", units="m"
    ) == pytest.approx(0.512, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:engine:width", units="m"
    ) == pytest.approx(0.512, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:engine:length", units="m"
    ) == pytest.approx(1.702, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_turboshaft_nacelle_dimensions():
    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftNacelleDimensions(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftNacelleDimensions(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:height", units="m"
    ) == pytest.approx(0.5632, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:width", units="m"
    ) == pytest.approx(0.5632, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:length", units="m"
    ) == pytest.approx(3.404, rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc_bis = get_indep_var_comp(
        list_inputs(
            SizingTurboshaftNacelleDimensions(turboshaft_id="turboshaft_1", position="in_the_front")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingTurboshaftNacelleDimensions(turboshaft_id="turboshaft_1", position="in_the_front"),
        ivc_bis,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:height", units="m"
    ) == pytest.approx(0.5632, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:width", units="m"
    ) == pytest.approx(0.5632, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:length", units="m"
    ) == pytest.approx(1.9573, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_nacelle_wet_area():
    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftNacelleWetArea(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftNacelleWetArea(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:wet_area", units="m**2"
    ) == pytest.approx(7.6685312, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_nacelle_drag():
    expected_drag_ls = [4.26, 0.0, 0.0]
    expected_drag_cruise = [4.217, 0.0, 0.0]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_drag_ls, expected_drag_cruise
    ):
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingTurboshaftDrag(
                        turboshaft_id="turboshaft_1", position=option, low_speed_aero=ls_option
                    )
                ),
                __file__,
                XML_FILE,
            )
            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingTurboshaftDrag(
                    turboshaft_id="turboshaft_1", position=option, low_speed_aero=ls_option
                ),
                ivc,
            )

            if ls_option:
                assert problem.get_val(
                    "data:propulsion:he_power_train:turboshaft:turboshaft_1:low_speed:CD0",
                ) * 1e3 == pytest.approx(ls_drag, rel=1e-2)
            else:
                assert problem.get_val(
                    "data:propulsion:he_power_train:turboshaft:turboshaft_1:cruise:CD0",
                ) * 1e3 == pytest.approx(cruise_drag, rel=1e-2)

            # Slight error on reynolds is due to step
            problem.check_partials(compact_print=True)


def test_turboshaft_cg_x():
    expected_cg = [3.703, 1.702, 3.679]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingTurboshaftCGX(turboshaft_id="turboshaft_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingTurboshaftCGX(turboshaft_id="turboshaft_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_turboshaft_cg_y():
    expected_cg = [1.7, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingTurboshaftCGY(turboshaft_id="turboshaft_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingTurboshaftCGY(turboshaft_id="turboshaft_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_turboshaft_sizing():
    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaft(turboshaft_id="turboshaft_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaft(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:mass", units="kg"
    ) == pytest.approx(231.57096, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:CG:x", units="m"
    ) == pytest.approx(3.703, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:CG:y", units="m"
    ) == pytest.approx(1.7, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:low_speed:CD0",
    ) * 1e3 == pytest.approx(4.273, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_density_ratio():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "density",
        val=np.linspace(1.225, 0.413, NB_POINTS_TEST),
        units="kg/m**3",
    )
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2000.0), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesDensityRatio(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("density_ratio") == pytest.approx(
        np.array([1.000, 0.926, 0.852, 0.779, 0.705, 0.631, 0.558, 0.484, 0.410, 0.337]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_mach_number():
    ivc = om.IndepVarComp()
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesMach(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("mach") == pytest.approx(
        np.array([0.240, 0.243, 0.246, 0.248, 0.251, 0.254, 0.257, 0.260, 0.263, 0.265]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_required_power():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesRequiredPower(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", val=np.linspace(250, 575.174, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesRequiredPower(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("power_required", units="kW") == pytest.approx(
        np.linspace(300, 625.174, NB_POINTS_TEST),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_power_for_power_rate():
    ivc = om.IndepVarComp()
    ivc.add_output("power_required", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPowerForPowerRate(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("shaft_power_for_power_rate", units="kW") == pytest.approx(
        np.linspace(300, 625.174, NB_POINTS_TEST),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_max_power_opr_limit():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesMaxPowerOPRLimit(
                turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "density_ratio",
        val=np.array([1.000, 0.926, 0.852, 0.779, 0.705, 0.631, 0.558, 0.484, 0.410, 0.337]),
        units="kg/m**3",
    )
    ivc.add_output(
        "mach",
        val=np.array([0.240, 0.243, 0.246, 0.248, 0.251, 0.254, 0.257, 0.260, 0.263, 0.265]),
    )
    ivc.add_output("power_required", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaxPowerOPRLimit(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("design_power_opr_limit", units="kW") == pytest.approx(
        np.array(
            [249.59, 308.87, 381.02, 469.77, 582.07, 726.97, 917.25, 1181.39, 1563.06, 2145.35]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_max_power_opr_limit_ref_point():
    # Same test as above but on a point close to the original model to see if it matches

    ivc = om.IndepVarComp()
    ivc.add_output("density_ratio", val=np.array([0.3813]), units="kg/m**3")
    ivc.add_output("mach", val=np.array([0.5]))
    ivc.add_output("power_required", val=np.array([446.32]), units="kW")
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:OPR", val=9.5
    )
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:T41t",
        val=1350,
        units="degK",
    )
    ivc.add_output("data:propulsion:he_power_train:turboshaft:turboshaft_1:limit:OPR", val=12.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaxPowerOPRLimit(turboshaft_id="turboshaft_1", number_of_points=1),
        ivc,
    )
    assert problem.get_val("design_power_opr_limit", units="kW") == pytest.approx(745.7, rel=1e-2)


def test_max_power_itt_limit():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesMaxPowerITTLimit(
                turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "density_ratio",
        val=np.array([1.000, 0.926, 0.852, 0.779, 0.705, 0.631, 0.558, 0.484, 0.410, 0.337]),
        units="kg/m**3",
    )
    ivc.add_output(
        "mach",
        val=np.array([0.240, 0.243, 0.246, 0.248, 0.251, 0.254, 0.257, 0.260, 0.263, 0.265]),
    )
    ivc.add_output("power_required", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaxPowerITTLimit(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("design_power_itt_limit", units="kW") == pytest.approx(
        np.array([250.26, 285.93, 324.67, 367.12, 414.49, 468.15, 529.82, 603.56, 694.52, 811.54]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_equivalent_power_itt_limit():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesEquivalentRatedPowerITTLimit(
                turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "design_power_itt_limit",
        units="kW",
        val=np.array(
            [250.26, 285.93, 324.67, 367.12, 414.49, 468.15, 529.82, 603.56, 694.52, 811.54]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEquivalentRatedPowerITTLimit(
            turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("equivalent_rated_power_itt_limit", units="kW") == pytest.approx(
        np.array([118.05, 134.87, 153.14, 173.17, 195.51, 220.82, 249.91, 284.70, 327.60, 382.80]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_equivalent_power_opr_limit():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesEquivalentRatedPowerOPRLimit(
                turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "design_power_opr_limit",
        units="kW",
        val=np.array(
            [249.59, 308.87, 381.02, 469.77, 582.07, 726.97, 917.25, 1181.39, 1563.06, 2145.35]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEquivalentRatedPowerOPRLimit(
            turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("equivalent_rated_power_opr_limit", units="kW") == pytest.approx(
        np.array([117.73, 145.69, 179.72, 221.58, 274.56, 342.91, 432.66, 557.25, 737.29, 1011.9]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()
    ivc.add_output("power_required", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")
    ivc.add_output(
        "equivalent_rated_power_opr_limit",
        val=np.array(
            [117.73, 145.69, 179.72, 221.58, 274.56, 342.91, 432.66, 557.25, 737.29, 1011.9]
        ),
        units="kW",
    )
    ivc.add_output(
        "equivalent_rated_power_itt_limit",
        val=np.array(
            [118.05, 134.87, 153.14, 173.17, 195.51, 220.82, 249.91, 284.70, 327.60, 382.80]
        ),
        units="kW",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_max", units="kW"
    ) == pytest.approx(
        1011.9,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumption():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTurboshaftFuelConsumption(
                turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "density_ratio",
        val=np.array([1.000, 0.926, 0.852, 0.779, 0.705, 0.631, 0.558, 0.484, 0.410, 0.337]),
        units="kg/m**3",
    )
    ivc.add_output(
        "mach",
        val=np.array([0.240, 0.243, 0.246, 0.248, 0.251, 0.254, 0.257, 0.260, 0.263, 0.265]),
    )
    ivc.add_output("power_required", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftFuelConsumption(
            turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("fuel_consumption", units="kg/h") == pytest.approx(
        np.array([212.15, 213.19, 213.01, 211.97, 209.76, 206.66, 202.97, 198.41, 193.29, 188.06]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumed():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumption",
        val=np.array(
            [212.15, 213.19, 213.01, 211.97, 209.76, 206.66, 202.97, 198.41, 193.29, 188.06]
        ),
        units="kg/h",
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftFuelConsumed(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([29.46, 29.60, 29.58, 29.44, 29.13, 28.70, 28.19, 27.55, 26.84, 26.11]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_sfc():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumption",
        val=np.array(
            [212.15, 213.19, 213.01, 211.97, 209.76, 206.66, 202.97, 198.41, 193.29, 188.06]
        ),
        units="kg/h",
    )
    ivc.add_output("power_required", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSFC(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("specific_fuel_consumption", units="kg/h/kW") == pytest.approx(
        np.array([0.707, 0.634, 0.572, 0.519, 0.471, 0.429, 0.392, 0.358, 0.328, 0.300]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_co_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([29.46, 29.60, 29.58, 29.44, 29.13, 28.70, 28.19, 27.55, 26.84, 26.11]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftInFlightCOEmissions(
            number_of_points=NB_POINTS_TEST, turboshaft_id="turboshaft_1"
        ),
        ivc,
    )

    assert problem.get_val("CO_emissions", units="g") == pytest.approx(
        np.array([147.3, 148.0, 147.9, 147.2, 145.65, 143.5, 140.95, 137.75, 134.2, 130.55]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_co2_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([29.46, 29.60, 29.58, 29.44, 29.13, 28.70, 28.19, 27.55, 26.84, 26.11]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftInFlightCO2Emissions(
            number_of_points=NB_POINTS_TEST, turboshaft_id="turboshaft_1"
        ),
        ivc,
    )

    assert problem.get_val("CO2_emissions", units="kg") == pytest.approx(
        np.array([92.9, 93.3, 93.3, 92.8, 91.9, 90.5, 88.9, 86.9, 84.6, 82.3]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_h2o_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([29.46, 29.60, 29.58, 29.44, 29.13, 28.70, 28.19, 27.55, 26.84, 26.11]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftInFlightH2OEmissions(
            number_of_points=NB_POINTS_TEST, turboshaft_id="turboshaft_1"
        ),
        ivc,
    )

    assert problem.get_val("H2O_emissions", units="kg") == pytest.approx(
        np.array([36.4, 36.6, 36.5, 36.4, 36.0, 35.5, 34.8, 34.0, 33.2, 32.2]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_sox_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([29.46, 29.60, 29.58, 29.44, 29.13, 28.70, 28.19, 27.55, 26.84, 26.11]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftInFlightSOxEmissions(
            number_of_points=NB_POINTS_TEST, turboshaft_id="turboshaft_1"
        ),
        ivc,
    )

    assert problem.get_val("SOx_emissions", units="g") == pytest.approx(
        np.array([23.568, 23.68, 23.664, 23.552, 23.304, 22.96, 22.552, 22.04, 21.472, 20.888]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_hc_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([29.46, 29.60, 29.58, 29.44, 29.13, 28.70, 28.19, 27.55, 26.84, 26.11]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftInFlightHCEmissions(
            number_of_points=NB_POINTS_TEST, turboshaft_id="turboshaft_1"
        ),
        ivc,
    )

    assert problem.get_val("HC_emissions", units="g") == pytest.approx(
        np.array([14.73, 14.8, 14.79, 14.72, 14.565, 14.35, 14.095, 13.775, 13.42, 13.055]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_nox_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([29.46, 29.60, 29.58, 29.44, 29.13, 28.70, 28.19, 27.55, 26.84, 26.11]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftInFlightNOxEmissions(
            number_of_points=NB_POINTS_TEST, turboshaft_id="turboshaft_1"
        ),
        ivc,
    )

    assert problem.get_val("NOx_emissions", units="g") == pytest.approx(
        np.array(
            [335.844, 337.44, 337.212, 335.616, 332.082, 327.18, 321.366, 314.07, 305.976, 297.654]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_in_flight_emissions_sum():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "CO2_emissions",
        units="kg",
        val=np.array([92.9, 93.3, 93.3, 92.8, 91.9, 90.5, 88.9, 86.9, 84.6, 82.3]),
    )
    ivc.add_output(
        "CO_emissions",
        units="g",
        val=np.array([147.3, 148.0, 147.9, 147.2, 145.65, 143.5, 140.95, 137.75, 134.2, 130.55]),
    )
    ivc.add_output(
        "NOx_emissions",
        units="g",
        val=np.array(
            [335.844, 337.44, 337.212, 335.616, 332.082, 327.18, 321.366, 314.07, 305.976, 297.654]
        ),
    )
    ivc.add_output(
        "SOx_emissions",
        units="g",
        val=np.array([23.568, 23.68, 23.664, 23.552, 23.304, 22.96, 22.552, 22.04, 21.472, 20.888]),
    )
    ivc.add_output(
        "H2O_emissions",
        units="kg",
        val=np.array([36.4, 36.6, 36.5, 36.4, 36.0, 35.5, 34.8, 34.0, 33.2, 32.2]),
    )
    ivc.add_output(
        "HC_emissions",
        units="g",
        val=np.array([14.73, 14.8, 14.79, 14.72, 14.565, 14.35, 14.095, 13.775, 13.42, 13.055]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftInFlightEmissionsSum(
            turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2",
        units="kg",
    ) == pytest.approx(897.4, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO",
        units="kg",
    ) == pytest.approx(1.423, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:NOx",
        units="kg",
    ) == pytest.approx(3.244, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:SOx",
        units="g",
    ) == pytest.approx(227.68, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:H2O",
        units="kg",
    ) == pytest.approx(351.6, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:HC",
        units="g",
    ) == pytest.approx(142.3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_in_flight_emissions():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumed_t",
        val=np.array([29.46, 29.60, 29.58, 29.44, 29.13, 28.70, 28.19, 27.55, 26.84, 26.11]),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaftInFlightEmissions(
            turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2",
        units="kg",
    ) == pytest.approx(897.4, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO",
        units="kg",
    ) == pytest.approx(1.423, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:NOx",
        units="kg",
    ) == pytest.approx(3.244, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:SOx",
        units="g",
    ) == pytest.approx(227.68, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:H2O",
        units="kg",
    ) == pytest.approx(351.6, rel=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:HC",
        units="g",
    ) == pytest.approx(142.3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances_turboshaft():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTurboshaft(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    # Unused but necessary for compatibility
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2000.0), units="min**-1")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("shaft_power_out", val=np.linspace(250, 575.174, NB_POINTS_TEST), units="kW")
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))
    ivc.add_output(
        "density",
        val=np.linspace(1.225, 0.413, NB_POINTS_TEST),
        units="kg/m**3",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTurboshaft(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([29.46, 29.60, 29.58, 29.44, 29.13, 28.70, 28.19, 27.55, 26.84, 26.11]),
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_max", units="kW"
    ) == pytest.approx(
        1011.9,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_slipstream_density_ratio():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "density",
        val=np.linspace(1.225, 0.413, NB_POINTS_TEST),
        units="kg/m**3",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SlipstreamDensityRatio(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("density_ratio") == pytest.approx(
        np.array([1.000, 0.926, 0.852, 0.779, 0.705, 0.631, 0.558, 0.484, 0.410, 0.337]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_slipstream_mach():
    ivc = om.IndepVarComp()
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SlipstreamMach(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("mach") == pytest.approx(
        np.array([0.240, 0.243, 0.246, 0.248, 0.251, 0.254, 0.257, 0.260, 0.263, 0.265]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_slipstream_required_power_bigger_power():
    # This component is meant to work in conjunction with the variable being outputed from the
    # mission so the shaft power will be longer than the other variables likes altitude and mach

    ivc = get_indep_var_comp(
        list_inputs(
            SlipstreamRequiredPower(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "shaft_power_out",
        val=np.concatenate(
            (np.full(1, 50.0), np.linspace(250, 575.174, NB_POINTS_TEST), np.full(1, 50.0))
        ),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamRequiredPower(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("power_required", units="kW") == pytest.approx(
        np.linspace(300, 625.174, NB_POINTS_TEST),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_exhaust_velocity():
    ivc = get_indep_var_comp(
        list_inputs(
            SlipstreamExhaustVelocity(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "density_ratio",
        val=np.array([1.000, 0.926, 0.852, 0.779, 0.705, 0.631, 0.558, 0.484, 0.410, 0.337]),
        units="kg/m**3",
    )
    ivc.add_output(
        "mach",
        val=np.array([0.240, 0.243, 0.246, 0.248, 0.251, 0.254, 0.257, 0.260, 0.263, 0.265]),
    )
    ivc.add_output("power_required", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamExhaustVelocity(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("exhaust_velocity", units="m/s") == pytest.approx(
        np.array([133.65, 139.96, 147.30, 155.90, 166.28, 179.01, 194.87, 215.66, 243.91, 284.17]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_exhaust_mass_flow():
    ivc = get_indep_var_comp(
        list_inputs(
            SlipstreamExhaustMassFlow(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "density_ratio",
        val=np.array([1.000, 0.926, 0.852, 0.779, 0.705, 0.631, 0.558, 0.484, 0.410, 0.337]),
        units="kg/m**3",
    )
    ivc.add_output("power_required", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamExhaustMassFlow(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("exhaust_mass_flow", units="kg/s") == pytest.approx(
        np.array([4.209, 4.084, 3.934, 3.764, 3.568, 3.349, 3.111, 2.847, 2.557, 2.245]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_exhaust_thrust():
    ivc = om.IndepVarComp()
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 300.0, NB_POINTS_TEST), units="m/s")
    ivc.add_output(
        "exhaust_velocity",
        val=np.array(
            [133.65, 139.96, 147.30, 155.90, 166.28, 179.01, 194.87, 215.66, 243.91, 284.17]
        ),
        units="m/s",
    )
    ivc.add_output(
        "exhaust_mass_flow",
        val=np.array([4.209, 4.084, 3.934, 3.764, 3.568, 3.349, 3.111, 2.847, 2.557, 2.245]),
        units="kg/s",
    )

    problem = run_system(
        SlipstreamExhaustThrust(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("exhaust_thrust", units="N") == pytest.approx(
        np.array([218.23665, 138.51112889, 66.92171111, 5.14413333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_delta_cd():
    ivc = get_indep_var_comp(
        list_inputs(SlipstreamTurboshaftDeltaCd(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "density",
        val=np.linspace(1.225, 0.413, NB_POINTS_TEST),
        units="kg/m**3",
    )
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("exhaust_thrust", val=np.linspace(250.0, 200.0, NB_POINTS_TEST), units="N")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamTurboshaftDeltaCd(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("delta_Cd") * 1e3 == pytest.approx(
        np.array([-8.584, -8.850, -9.181, -9.592, -10.10, -10.76, -11.62, -12.76, -14.34, -16.64]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_slipstream():
    ivc = get_indep_var_comp(
        list_inputs(
            SlipstreamTurboshaft(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    # Fake variable at beginning and end required because ... ... that's why
    ivc.add_output(
        "shaft_power_out",
        val=np.concatenate(
            (np.full(1, 50.0), np.linspace(250, 575.174, NB_POINTS_TEST), np.full(1, 50.0))
        ),
        units="kW",
    )
    ivc.add_output(
        "density",
        val=np.linspace(1.225, 0.413, NB_POINTS_TEST),
        units="kg/m**3",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamTurboshaft(turboshaft_id="turboshaft_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("delta_Cd") * 1e3 == pytest.approx(
        np.array([-7.495, -8.460, -9.614, -11.01, -12.76, -14.99, -17.90, -21.87, -27.54, -36.18]),
        rel=1e-2,
    )
    assert problem.get_val("delta_Cl") == pytest.approx(np.zeros(NB_POINTS_TEST), rel=1e-2)
    assert problem.get_val("delta_Cm") == pytest.approx(np.zeros(NB_POINTS_TEST), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:mass",
        "data:environmental_impact:aircraft_per_fu",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCATurboshaftProdWeightPerFU(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:mass_per_fu", units="kg"
    ) == pytest.approx(0.00023157, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_emissions_per_fu():
    inputs_list = [
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2",
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO",
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:NOx",
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:SOx",
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:HC",
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:H2O",
        "data:environmental_impact:flight_per_fu",
        "data:environmental_impact:aircraft_per_fu",
        "data:environmental_impact:line_test:mission_ratio",
        "data:environmental_impact:delivery:mission_ratio",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCATurboshaftUseEmissionPerFU(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:LCA:operation:he_power_train:turboshaft:turboshaft_1:CO2_per_fu",
        units="kg",
    ) == pytest.approx(2.52640617, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:turboshaft:turboshaft_1:CO_per_fu",
        units="kg",
    ) == pytest.approx(0.00400381, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:turboshaft:turboshaft_1:NOx_per_fu",
        units="kg",
    ) == pytest.approx(0.00912869, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:turboshaft:turboshaft_1:SOx_per_fu",
        units="kg",
    ) == pytest.approx(0.00064061, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:turboshaft:turboshaft_1:H2O_per_fu",
        units="kg",
    ) == pytest.approx(0.9905434, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:turboshaft:turboshaft_1:HC_per_fu",
        units="kg",
    ) == pytest.approx(0.00040038, rel=1e-3)

    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:turboshaft:turboshaft_1:CO2_per_fu", units="kg"
    ) == pytest.approx(0.01894805, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:turboshaft:turboshaft_1:CO_per_fu", units="kg"
    ) == pytest.approx(3.0028574999999997e-05, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:turboshaft:turboshaft_1:NOx_per_fu", units="kg"
    ) == pytest.approx(6.846517500000001e-05, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:turboshaft:turboshaft_1:SOx_per_fu", units="kg"
    ) == pytest.approx(4.80457591e-06, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:turboshaft:turboshaft_1:H2O_per_fu", units="kg"
    ) == pytest.approx(0.0074290755, rel=1e-3)
    assert problem.get_val(
        "data:LCA:manufacturing:he_power_train:turboshaft:turboshaft_1:HC_per_fu", units="kg"
    ) == pytest.approx(3.0028499999999995e-06, rel=1e-3)

    assert problem.get_val(
        "data:LCA:distribution:he_power_train:turboshaft:turboshaft_1:CO2_per_fu", units="kg"
    ) == pytest.approx(0.01263203, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:turboshaft:turboshaft_1:CO_per_fu", units="kg"
    ) == pytest.approx(2.00190663e-05, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:turboshaft:turboshaft_1:NOx_per_fu", units="kg"
    ) == pytest.approx(4.56434712e-05, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:turboshaft:turboshaft_1:SOx_per_fu", units="kg"
    ) == pytest.approx(3.20305061e-06, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:turboshaft:turboshaft_1:H2O_per_fu", units="kg"
    ) == pytest.approx(0.00495272, rel=1e-3)
    assert problem.get_val(
        "data:LCA:distribution:he_power_train:turboshaft:turboshaft_1:HC_per_fu", units="kg"
    ) == pytest.approx(2.00190663e-06, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cost():
    ivc = om.IndepVarComp()
    ivc.add_output("data:cost:cpi_2012", val=1.4)
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating",
        val=575.174,
        units="kW",
    )

    problem = run_system(
        LCCTurboshaftCost(turboshaft_id="turboshaft_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:purchase_cost", units="USD"
    ) == pytest.approx(
        407535.115,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_operational_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating",
        val=575.174,
        units="kW",
    )

    problem = run_system(
        LCCTurboshaftOperationalCost(turboshaft_id="turboshaft_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:operational_cost",
        units="USD/yr",
    ) == pytest.approx(
        30357.83,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)
