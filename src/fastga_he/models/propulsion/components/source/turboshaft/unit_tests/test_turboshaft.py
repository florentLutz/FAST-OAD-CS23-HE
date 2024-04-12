# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

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

from ..components.perf_density_ratio import PerformancesDensityRatio
from ..components.perf_mach import PerformancesMach
from ..components.perf_max_power_opr_limit import PerformancesMaxPowerOPRLimit
from ..components.perf_max_power_itt_limit import PerformancesMaxPowerITTLimit
from ..components.perf_equivalent_rated_power_itt_limit import (
    PerformancesEquivalentRatedPowerITTLimit,
)
from ..components.perf_equivalent_rated_power_opr_limit import (
    PerformancesEquivalentRatedPowerOPRLimit,
)
from ..components.perf_maximum import PerformancesMaximum

from ..components.sizing_turboshaft import SizingTurboshaft

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_turboshaft.xml"
NB_POINTS_TEST = 10


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
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:turboshaft:turboshaft_1:low_speed:CD0",
                    )
                    * 1e3
                    == pytest.approx(ls_drag, rel=1e-2)
                )
            else:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:turboshaft:turboshaft_1:cruise:CD0",
                    )
                    * 1e3
                    == pytest.approx(cruise_drag, rel=1e-2)
                )

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


def test_ice_sizing():

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
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:low_speed:CD0",
        )
        * 1e3
        == pytest.approx(4.273, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_density_ratio():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "density",
        val=np.linspace(1.225, 0.413, NB_POINTS_TEST),
        units="kg/m**3",
    )

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
    ivc.add_output("shaft_power_out", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")

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
    ivc.add_output("shaft_power_out", val=np.array([446.32]), units="kW")
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
    ivc.add_output("shaft_power_out", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")

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
    ivc.add_output("shaft_power_out", val=np.linspace(300, 625.174, NB_POINTS_TEST), units="kW")
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
