# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import pytest

import copy

import openmdao.api as om

from ..components.perf_rpm_in import PerformancesRPMIn
from ..components.perf_mission_power_split import PerformancesMissionPowerSplit
from ..components.perf_mission_power_share import PerformancesMissionPowerShare
from ..components.perf_percent_split_equivalent import PerformancesPercentSplitEquivalent
from ..components.perf_shaft_power_in import PerformancesShaftPowerIn
from ..components.perf_torque_out import PerformancesTorqueOut
from ..components.perf_torque_in import PerformancesTorqueIn
from ..components.perf_maximum import PerformancesMaximum

from ..components.cstr_enforce import ConstraintsTorqueEnforce
from ..components.cstr_ensure import ConstraintsTorqueEnsure

from ..components.sizing_weight import SizingPlanetaryGearWeight
from ..components.sizing_dimension_scaling import SizingPlanetaryGearDimensionScaling
from ..components.sizing_dimension import SizingPlanetaryGearDimensions
from ..components.sizing_cg_x import SizingPlanetaryGearCGX
from ..components.sizing_cg_y import SizingPlanetaryGearCGY

from ..components.pre_lca_prod_weight_per_fu import PreLCAPlanetaryGearProdWeightPerFU
from ..components.lcc_planatary_gear_cost import LCCPlanetaryGearCost

from ..components.perf_planetary_gear import PerformancesPlanetaryGear
from ..components.sizing_planetary_gear import SizingPlanetaryGear

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_planetary_gear.xml"
NB_POINTS_TEST = 10


def test_rpm_in():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesRPMIn(planetary_gear_id="planetary_gear_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm_out", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesRPMIn(planetary_gear_id="planetary_gear_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("rpm_in_1", units="min**-1") == pytest.approx(
        np.linspace(2000.0, 2500.0, NB_POINTS_TEST), rel=1e-2
    )

    assert problem.get_val("rpm_in_1", units="s**-1") == pytest.approx(
        problem.get_val("rpm_in_2", units="s**-1"), rel=1e-8
    )

    problem.check_partials(compact_print=True)


def test_perf_power_split_formatting():
    power_split_float = 42.0
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        val=power_split_float,
        units="percent",
    )

    problem = run_system(
        PerformancesMissionPowerSplit(
            planetary_gear_id="planetary_gear_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    power_split_output = problem.get_val(
        "power_split",
        units="percent",
    )
    expected_power_split = np.full(NB_POINTS_TEST, power_split_float)
    assert power_split_output == pytest.approx(expected_power_split, rel=1e-4)

    problem.check_partials(compact_print=True)

    # Let's now try with a full power split
    power_split_array = np.linspace(60, 40, NB_POINTS_TEST)
    ivc_2 = om.IndepVarComp()
    ivc_2.add_output(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_split",
        val=power_split_array,
        units="percent",
    )

    problem = run_system(
        PerformancesMissionPowerSplit(
            planetary_gear_id="planetary_gear_1", number_of_points=NB_POINTS_TEST
        ),
        ivc_2,
    )
    power_split_output = problem.get_val(
        "power_split",
        units="percent",
    )
    assert power_split_output == pytest.approx(power_split_array, rel=1e-4)

    problem.check_partials(compact_print=True)


def test_perf_power_share_formatting():
    power_split_float = 150.0e3
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_share",
        val=power_split_float,
        units="W",
    )

    problem = run_system(
        PerformancesMissionPowerShare(
            planetary_gear_id="planetary_gear_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    power_split_output = problem.get_val(
        "power_share",
        units="W",
    )
    expected_power_split = np.full(NB_POINTS_TEST, power_split_float)
    assert power_split_output == pytest.approx(expected_power_split, rel=1e-4)

    problem.check_partials(compact_print=True)

    # Let's now try with a full power split
    power_split_array = np.linspace(60e3, 40e3, NB_POINTS_TEST)
    ivc_2 = om.IndepVarComp()
    ivc_2.add_output(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:power_share",
        val=power_split_array,
        units="W",
    )

    problem = run_system(
        PerformancesMissionPowerShare(
            planetary_gear_id="planetary_gear_1", number_of_points=NB_POINTS_TEST
        ),
        ivc_2,
    )
    power_split_output = problem.get_val(
        "power_share",
        units="W",
    )
    assert power_split_output == pytest.approx(power_split_array, rel=1e-4)

    problem.check_partials(compact_print=True)


def test_percent_split_equivalent():
    ivc_orig = get_indep_var_comp(
        list_inputs(
            PerformancesPercentSplitEquivalent(
                planetary_gear_id="planetary_gear_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc_orig.add_output("shaft_power_out", val=np.full(NB_POINTS_TEST, 250), units="kW")

    # The requirement for the primary branch is below the output
    ivc_low_req = copy.deepcopy(ivc_orig)
    ivc_low_req.add_output("power_share", val=np.full(NB_POINTS_TEST, 200 / 0.98), units="kW")

    problem_low_req = run_system(
        PerformancesPercentSplitEquivalent(
            number_of_points=NB_POINTS_TEST, planetary_gear_id="planetary_gear_1"
        ),
        ivc_low_req,
    )

    assert problem_low_req.get_val("power_split", units="percent") == pytest.approx(
        np.full(NB_POINTS_TEST, 80.0), rel=1e-4
    )

    problem_low_req.check_partials(compact_print=True)

    # The requirement for the primary branch is above the output
    ivc_high_req = copy.deepcopy(ivc_orig)
    ivc_high_req.add_output("power_share", val=np.full(NB_POINTS_TEST, 300), units="kW")

    problem_high_req = run_system(
        PerformancesPercentSplitEquivalent(
            number_of_points=NB_POINTS_TEST, planetary_gear_id="planetary_gear_1"
        ),
        ivc_high_req,
    )

    assert problem_high_req.get_val("power_split", units="percent") == pytest.approx(
        np.full(NB_POINTS_TEST, 100.0), rel=1e-4
    )

    problem_high_req.check_partials(compact_print=True)

    # The requirement for the primary branch is above the output
    ivc_low_to_high_req = copy.deepcopy(ivc_orig)
    ivc_low_to_high_req.add_output(
        "power_share", val=np.linspace(200, 300, NB_POINTS_TEST) / 0.98, units="kW"
    )

    problem_low_to_high_req = run_system(
        PerformancesPercentSplitEquivalent(
            number_of_points=NB_POINTS_TEST, planetary_gear_id="planetary_gear_1"
        ),
        ivc_low_to_high_req,
    )

    assert problem_low_to_high_req.get_val("power_split", units="percent") == pytest.approx(
        np.array([80.0, 84.444, 88.888, 93.333, 97.777, 100.0, 100.0, 100.0, 100.0, 100.0]),
        rel=1e-4,
    )

    problem_low_to_high_req.check_partials(compact_print=True)


def test_shaft_power_in():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesShaftPowerIn(
                planetary_gear_id="planetary_gear_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")
    ivc.add_output(
        "power_split",
        val=np.array([80.0, 84.444, 88.888, 93.333, 97.777, 100.0, 100.0, 100.0, 100.0, 100.0]),
        units="percent",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesShaftPowerIn(
            planetary_gear_id="planetary_gear_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("shaft_power_in_1", units="kW") == pytest.approx(
        np.array([81.63, 95.74, 110.8, 126.9, 144.1, 158.7, 170.0, 181.4, 192.7, 204.0]),
        rel=1e-3,
    )
    assert problem.get_val("shaft_power_in_2", units="kW") == pytest.approx(
        np.array([20.408, 17.637, 13.8585, 9.0707, 3.2765, 0.0, 0.0, 0.0, 0.0, 0.0]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_torque_out():
    ivc = om.IndepVarComp()
    ivc.add_output("shaft_power_out", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm_out", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorqueOut(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_out", units="N*m") == pytest.approx(
        np.array([477.4, 516.1, 552.8, 587.6, 620.7, 652.1, 682.0, 710.6, 737.9, 763.9]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_torque_in():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_in_1",
        val=np.array([81.63, 95.74, 110.8, 126.9, 144.1, 158.7, 170.0, 181.4, 192.7, 204.0]),
        units="kW",
    )
    ivc.add_output(
        "shaft_power_in_2",
        val=np.array([20.408, 17.637, 13.8585, 9.0707, 3.2765, 0.0, 0.0, 0.0, 0.0, 0.0]),
        units="kW",
    )
    ivc.add_output("rpm_in_1", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")
    ivc.add_output("rpm_in_2", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorqueIn(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_in_1", units="N*m") == pytest.approx(
        np.array([389.7, 444.7, 501.1, 559.2, 619.2, 665.3, 695.7, 725.1, 752.7, 779.2]),
        rel=1e-3,
    )
    assert problem.get_val("torque_in_2", units="N*m") == pytest.approx(
        np.array([97.441, 81.934, 62.686, 39.977, 14.079, 0.0, 0.0, 0.0, 0.0, 0.0]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "torque_out",
        val=np.array([477.4, 516.1, 552.8, 587.6, 620.7, 652.1, 682.0, 710.6, 737.9, 763.9]),
        units="N*m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(number_of_points=NB_POINTS_TEST, planetary_gear_id="planetary_gear_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:torque_out_max", units="N*m"
    ) == pytest.approx(763.9, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_performances_planetary_gear_percent_split():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesPlanetaryGear(
                planetary_gear_id="planetary_gear_1",
                number_of_points=NB_POINTS_TEST,
                gear_mode="percent_split",
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm_out", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")
    ivc.add_output("shaft_power_out", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPlanetaryGear(
            planetary_gear_id="planetary_gear_1",
            number_of_points=NB_POINTS_TEST,
            gear_mode="percent_split",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:torque_out_max", units="N*m"
    ) == pytest.approx(763.9, rel=1e-3)
    assert problem.get_val("shaft_power_in_1", units="kW") == pytest.approx(
        np.array([68.02, 75.57, 83.13, 90.69, 98.25, 105.8, 113.3, 120.9, 128.4, 136.0]),
        rel=1e-3,
    )
    assert problem.get_val("torque_in_1", units="N*m") == pytest.approx(
        2.0 * problem.get_val("torque_in_2", units="N*m"), rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_performances_planetary_gear_power_share():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesPlanetaryGear(
                planetary_gear_id="planetary_gear_1",
                number_of_points=NB_POINTS_TEST,
                gear_mode="power_share",
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm_out", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")
    ivc.add_output("shaft_power_out", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPlanetaryGear(
            planetary_gear_id="planetary_gear_1",
            number_of_points=NB_POINTS_TEST,
            gear_mode="power_share",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:torque_out_max", units="N*m"
    ) == pytest.approx(763.9, rel=1e-3)
    assert problem.get_val("shaft_power_in_1", units="kW") == pytest.approx(
        np.full(NB_POINTS_TEST, 66.66),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_torque_constraint_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnforce(planetary_gear_id="planetary_gear_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnforce(planetary_gear_id="planetary_gear_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:torque_out_rating",
        units="N*m",
    ) == pytest.approx(790.0, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_torque_constraint_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnsure(planetary_gear_id="planetary_gear_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnsure(planetary_gear_id="planetary_gear_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:planetary_gear:planetary_gear_1:torque_out_rating",
        units="N*m",
    ) == pytest.approx(40.0, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_sizing_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingPlanetaryGearWeight(planetary_gear_id="planetary_gear_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPlanetaryGearWeight(planetary_gear_id="planetary_gear_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:mass", units="kg"
    ) == pytest.approx(15.59, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_dimension_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingPlanetaryGearDimensionScaling(planetary_gear_id="planetary_gear_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPlanetaryGearDimensionScaling(planetary_gear_id="planetary_gear_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:scaling:dimensions"
    ) == pytest.approx(1.34, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_dimension():
    ivc = get_indep_var_comp(
        list_inputs(SizingPlanetaryGearDimensions(planetary_gear_id="planetary_gear_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPlanetaryGearDimensions(planetary_gear_id="planetary_gear_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:width", units="m"
    ) == pytest.approx(0.201, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:height", units="m"
    ) == pytest.approx(0.201, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:length", units="m"
    ) == pytest.approx(0.3551, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_cg_x():
    expected_cg = [2.69, 0.45, 2.54]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingPlanetaryGearCGX(planetary_gear_id="planetary_gear_1", position=option)
            ),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingPlanetaryGearCGX(planetary_gear_id="planetary_gear_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:CG:x",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_cg_y():
    expected_cg = [2.21, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingPlanetaryGearCGY(planetary_gear_id="planetary_gear_1", position=option)
            ),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingPlanetaryGearCGY(planetary_gear_id="planetary_gear_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:CG:y",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_planetary_gear_sizing():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPlanetaryGear(planetary_gear_id="planetary_gear_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingPlanetaryGear(planetary_gear_id="planetary_gear_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:mass",
        units="kg",
    ) == pytest.approx(16.22, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:CG:x", units="m"
    ) == pytest.approx(2.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:CG:y", units="m"
    ) == pytest.approx(2.21, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:low_speed:CD0",
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:cruise:CD0",
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:mass",
        "data:environmental_impact:aircraft_per_fu",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PreLCAPlanetaryGearProdWeightPerFU(planetary_gear_id="planetary_gear_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:mass_per_fu", units="kg"
    ) == pytest.approx(1.622e-05, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:mass", val=32.86, units="kg"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCPlanetaryGearCost(planetary_gear_id="planetary_gear_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:planetary_gear:planetary_gear_1:cost_per_unit", units="USD"
    ) == pytest.approx(7590.0, rel=1e-2)

    problem.check_partials(compact_print=True)
