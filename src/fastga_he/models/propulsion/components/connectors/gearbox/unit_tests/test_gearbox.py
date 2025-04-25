# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.perf_rpm_in import PerformancesRPMIn
from ..components.perf_shaft_power_in import PerformancesShaftPowerIn
from ..components.perf_torque_out import PerformancesTorqueOut
from ..components.perf_torque_in import PerformancesTorqueIn
from ..components.perf_maximum import PerformancesMaximum

from ..components.cstr_enforce import ConstraintsTorqueEnforce
from ..components.cstr_ensure import ConstraintsTorqueEnsure

from ..components.sizing_weight import SizingGearboxWeight
from ..components.sizing_dimension_scaling import SizingGearboxDimensionScaling
from ..components.sizing_dimension import SizingGearboxDimensions
from ..components.sizing_cg_x import SizingGearboxCGX
from ..components.sizing_cg_y import SizingGearboxCGY

from ..components.pre_lca_prod_weight_per_fu import PreLCAGearboxProdWeightPerFU
from ..components.lcc_gearbox_cost import LCCGearboxCost

from ..components.perf_gearbox import PerformancesGearbox
from ..components.sizing_gearbox import SizingGearbox

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_gearbox.xml"
NB_POINTS_TEST = 10


def test_rpm_in():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesRPMIn(gearbox_id="gearbox_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm_out_1", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesRPMIn(gearbox_id="gearbox_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("rpm_in", units="min**-1") == pytest.approx(
        np.linspace(2000.0, 2500.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_shaft_power_in():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesShaftPowerIn(gearbox_id="gearbox_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out_1", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")
    ivc.add_output("shaft_power_out_2", val=np.linspace(200.0, 150.0, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesShaftPowerIn(gearbox_id="gearbox_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.linspace(306.12, 357.14, NB_POINTS_TEST),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_torque_out():
    ivc = om.IndepVarComp()
    ivc.add_output("shaft_power_out_1", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")
    ivc.add_output("shaft_power_out_2", val=np.linspace(200.0, 150.0, NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm_out_1", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")
    ivc.add_output("rpm_out_2", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorqueOut(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_out_1", units="N*m") == pytest.approx(
        np.array([477.4, 516.1, 552.8, 587.6, 620.7, 652.1, 682.0, 710.6, 737.9, 763.9]),
        rel=1e-3,
    )
    assert problem.get_val("torque_out_2", units="N*m") == pytest.approx(
        np.array([954.9, 903.3, 854.4, 808.0, 763.9, 722.0, 682.0, 644.0, 607.6, 572.9]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_torque_in():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_in",
        val=np.linspace(306.12, 357.14, NB_POINTS_TEST),
        units="kW",
    )
    ivc.add_output("rpm_in", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorqueIn(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_in", units="N*m") == pytest.approx(
        np.array([1461.6, 1448.4, 1435.9, 1424.1, 1412.8, 1402.2, 1392.0, 1382.3, 1373.0, 1364.1]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "torque_out_1",
        val=np.array([477.4, 516.1, 552.8, 587.6, 620.7, 652.1, 682.0, 710.6, 737.9, 763.9]),
        units="N*m",
    )
    ivc.add_output(
        "torque_out_2",
        val=np.array([954.9, 903.3, 854.4, 808.0, 763.9, 722.0, 682.0, 644.0, 607.6, 572.9]),
        units="N*m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(number_of_points=NB_POINTS_TEST, gearbox_id="gearbox_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:torque_out_max", units="N*m"
    ) == pytest.approx(1432.3, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_performances_gearbox():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesGearbox(
                gearbox_id="gearbox_1",
                number_of_points=NB_POINTS_TEST,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm_out_1", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")
    ivc.add_output("rpm_out_2", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")
    ivc.add_output("shaft_power_out_1", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")
    ivc.add_output("shaft_power_out_2", val=np.linspace(300.0, 290.0, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesGearbox(
            gearbox_id="gearbox_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:torque_out_max", units="N*m"
    ) == pytest.approx(1909.0, rel=1e-3)
    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.linspace(408.0, 500.0, NB_POINTS_TEST),
        rel=1e-3,
    )
    assert problem.get_val("torque_in", units="N*m") == pytest.approx(
        np.linspace(1948.0, 1909.0, NB_POINTS_TEST), rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_torque_constraint_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnforce(gearbox_id="gearbox_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnforce(gearbox_id="gearbox_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:torque_out_rating",
        units="N*m",
    ) == pytest.approx(1900.0, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_torque_constraint_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnsure(gearbox_id="gearbox_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnsure(gearbox_id="gearbox_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:gearbox:gearbox_1:torque_out_rating",
        units="N*m",
    ) == pytest.approx(-100.0, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_sizing_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingGearboxWeight(gearbox_id="gearbox_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGearboxWeight(gearbox_id="gearbox_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:mass", units="kg"
    ) == pytest.approx(32.86, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_dimension_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingGearboxDimensionScaling(gearbox_id="gearbox_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGearboxDimensionScaling(gearbox_id="gearbox_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:scaling:dimensions"
    ) == pytest.approx(1.86, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_dimension():
    ivc = get_indep_var_comp(
        list_inputs(SizingGearboxDimensions(gearbox_id="gearbox_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGearboxDimensions(gearbox_id="gearbox_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:width", units="m"
    ) == pytest.approx(0.279, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:height", units="m"
    ) == pytest.approx(0.279, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:length", units="m"
    ) == pytest.approx(0.4929, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_cg_x():
    expected_cg = [2.69, 0.4, 3.10]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingGearboxCGX(gearbox_id="gearbox_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingGearboxCGX(gearbox_id="gearbox_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:gearbox:gearbox_1:CG:x",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_cg_y():
    expected_cg = [1.3, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingGearboxCGY(gearbox_id="gearbox_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingGearboxCGY(gearbox_id="gearbox_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:gearbox:gearbox_1:CG:y",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_gearbox_sizing():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingGearbox(gearbox_id="gearbox_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingGearbox(gearbox_id="gearbox_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:mass",
        units="kg",
    ) == pytest.approx(31.6, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:CG:x", units="m"
    ) == pytest.approx(2.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:CG:y", units="m"
    ) == pytest.approx(1.3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:low_speed:CD0",
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:cruise:CD0",
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:gearbox:gearbox_1:mass",
        "data:environmental_impact:aircraft_per_fu",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCAGearboxProdWeightPerFU(gearbox_id="gearbox_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:mass_per_fu", units="kg"
    ) == pytest.approx(3.286e-05, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cost():
    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:he_power_train:gearbox:gearbox_1:mass", val=32.86, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCGearboxCost(gearbox_id="gearbox_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:cost_per_unit", units="USD"
    ) == pytest.approx(7590.0, rel=1e-2)

    problem.check_partials(compact_print=True)
