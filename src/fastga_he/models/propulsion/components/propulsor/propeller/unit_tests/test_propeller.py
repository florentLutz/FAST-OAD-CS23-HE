# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.sizing_weight import SizingPropellerWeight
from ..components.perf_mission_rpm import PerformancesRPMMission
from ..components.perf_advance_ratio import PerformancesAdvanceRatio
from ..components.perf_tip_mach import PerformancesTipMach
from ..components.perf_thrust_coefficient import PerformancesThrustCoefficient
from ..components.perf_blade_reynolds import PerformancesBladeReynoldsNumber
from ..components.perf_power_coefficient import PerformancesPowerCoefficient
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_shaft_power import PerformancesShaftPower
from ..components.perf_torque import PerformancesTorque
from ..components.perf_maximum import PerformancesMaximum
from ..components.cstr_enforce import ConstraintsTorqueEnforce
from ..components.cstr_ensure import ConstraintsTorqueEnsure

from ..components.perf_propeller import PerformancesPropeller
from ..components.sizing_propeller import SizingPropeller

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_propeller.xml"
NB_POINTS_TEST = 10


def test_weight():

    ivc = get_indep_var_comp(
        list_inputs(SizingPropellerWeight(propeller_id="propeller_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPropellerWeight(propeller_id="propeller_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:mass", units="lbm"
    ) == pytest.approx(
        80.1, rel=1e-2
    )  # Real value for Cirrus SR22 prop is 75 lbs

    problem.check_partials(compact_print=True)


def test_constraints_torque_enforce():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnforce(propeller_id="propeller_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnforce(propeller_id="propeller_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:torque_rating", units="N*m"
    ) == pytest.approx(817.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_torque_ensure():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnsure(propeller_id="propeller_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnsure(propeller_id="propeller_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:propeller:propeller_1:torque_rating", units="N*m"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rpm_mission():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:propeller:propeller_1:rpm_mission",
        val=2500.0,
        units="min**-1",
    )

    problem = run_system(
        PerformancesRPMMission(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("rpm", units="min**-1") == pytest.approx(
        np.full(NB_POINTS_TEST, 2500), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:propeller:propeller_1:rpm_mission",
        val=[2700, 2500, 2000, 2700, 2500, 2000, 2700, 2500, 2000, 420],
        units="min**-1",
    )

    problem3 = run_system(
        PerformancesRPMMission(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST), ivc3
    )

    assert problem3.get_val("rpm", units="min**-1") == pytest.approx(
        np.array([2700, 2500, 2000, 2700, 2500, 2000, 2700, 2500, 2000, 420]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_advance_ratio():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesAdvanceRatio(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500), units="min**-1")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAdvanceRatio(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("advance_ratio") == pytest.approx(
        np.array([0.99, 1.0, 1.01, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.1]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_tip_mach():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTipMach(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500), units="min**-1")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTipMach(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("tip_mach") == pytest.approx(
        np.array([0.639, 0.64, 0.641, 0.643, 0.644, 0.646, 0.647, 0.649, 0.65, 0.652]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_thrust_coefficient():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesThrustCoefficient(
                propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesThrustCoefficient(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("thrust_coefficient") == pytest.approx(
        np.array([0.0473, 0.047, 0.0466, 0.0463, 0.0459, 0.0456, 0.0453, 0.0449, 0.0446, 0.0443]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_blade_reynolds_number():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesBladeReynoldsNumber(
                propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500), units="min**-1")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesBladeReynoldsNumber(
            propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert problem.get_val("reynolds_D") == pytest.approx(
        np.array(
            [
                36882132.0,
                36921782.0,
                36961855.0,
                37002349.0,
                37043262.0,
                37084595.0,
                37126344.0,
                37168508.0,
                37211087.0,
                37254079.0,
            ]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_power_coefficient():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesPowerCoefficient(
                propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "thrust_coefficient",
        val=np.array(
            [0.0473, 0.047, 0.0466, 0.0463, 0.0459, 0.0456, 0.0453, 0.0449, 0.0446, 0.0443]
        ),
    )
    ivc.add_output(
        "advance_ratio",
        val=np.array([0.99, 1.0, 1.01, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.1]),
    )
    ivc.add_output(
        "tip_mach",
        val=np.array([0.639, 0.64, 0.641, 0.643, 0.644, 0.646, 0.647, 0.649, 0.65, 0.652]),
    )
    ivc.add_output(
        "reynolds_D",
        val=np.array(
            [
                36882132.0,
                36921782.0,
                36961855.0,
                37002349.0,
                37043262.0,
                37084595.0,
                37126344.0,
                37168508.0,
                37211087.0,
                37254079.0,
            ]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPowerCoefficient(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("power_coefficient") == pytest.approx(
        np.array([0.0658, 0.0661, 0.0662, 0.0671, 0.0672, 0.0675, 0.0677, 0.0678, 0.0681, 0.0689]),
        rel=1e-2,
    )

    # Derivative wrt Re is accurate with the proper step (at least 1)
    problem.check_partials(compact_print=True)


def test_efficiency():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesEfficiency(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "thrust_coefficient",
        val=np.array(
            [0.0473, 0.047, 0.0466, 0.0463, 0.0459, 0.0456, 0.0453, 0.0449, 0.0446, 0.0443]
        ),
    )
    ivc.add_output(
        "power_coefficient",
        val=np.array(
            [0.0658, 0.0661, 0.0662, 0.0671, 0.0672, 0.0675, 0.0677, 0.0678, 0.0681, 0.0689]
        ),
    )
    ivc.add_output(
        "advance_ratio",
        val=np.array([0.99, 1.0, 1.01, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.1]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEfficiency(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("efficiency") == pytest.approx(
        np.array([0.712, 0.711, 0.711, 0.711, 0.71, 0.709, 0.709, 0.709, 0.707, 0.707]),
        rel=1e-2,
    )

    # Derivative wrt Re is accurate with the proper step (at least 1)
    problem.check_partials(compact_print=True)


def test_shaft_power():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesShaftPower(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500), units="min**-1")
    ivc.add_output(
        "power_coefficient",
        val=np.array(
            [0.0658, 0.0661, 0.0662, 0.0671, 0.0672, 0.0675, 0.0677, 0.0678, 0.0681, 0.0689]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesShaftPower(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.array([178.0, 178.8, 179.1, 181.5, 181.8, 182.6, 183.1, 183.4, 184.2, 186.4]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_torque():

    ivc = om.IndepVarComp()
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500), units="min**-1")
    ivc.add_output(
        "shaft_power_in",
        val=np.array([178.0, 178.8, 179.1, 181.5, 181.8, 182.6, 183.1, 183.4, 184.2, 186.4]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTorque(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("torque_in", units="N*m") == pytest.approx(
        np.array([679.9, 683.0, 684.1, 693.3, 694.4, 697.5, 699.4, 700.5, 703.6, 712.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500), units="min**-1")
    ivc.add_output(
        "torque_in",
        val=np.array([679.9, 683.0, 684.1, 693.3, 694.4, 697.5, 699.4, 700.5, 703.6, 712.0]),
        units="N*m",
    )
    ivc.add_output(
        "advance_ratio",
        val=np.array([0.99, 1.0, 1.01, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.1]),
    )
    ivc.add_output(
        "tip_mach",
        val=np.array([0.639, 0.64, 0.641, 0.643, 0.644, 0.646, 0.647, 0.649, 0.65, 0.652]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:tip_mach_max"
    ) == pytest.approx(0.652, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:advance_ratio_max"
    ) == pytest.approx(1.1, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:torque_max", units="N*m"
    ) == pytest.approx(712.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:rpm_max", units="min**-1"
    ) == pytest.approx(2500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_propeller():

    ivc = get_indep_var_comp(
        list_inputs(SizingPropeller(propeller_id="propeller_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPropeller(propeller_id="propeller_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:mass", units="lbm"
    ) == pytest.approx(
        80.1, rel=1e-2
    )  # Real value for Cirrus SR22 prop is 75 lbs

    problem.check_partials(compact_print=True)


def test_propeller_performances():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesPropeller(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPropeller(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.array([178.0, 178.8, 179.1, 181.5, 181.8, 182.6, 183.1, 183.4, 184.2, 186.4]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)
