# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.cstr_enforce import (
    ConstraintsTorqueEnforce,
    ConstraintsRPMEnforce,
    ConstraintsVoltageEnforce,
)
from ..components.cstr_ensure import (
    ConstraintsTorqueEnsure,
    ConstraintsRPMEnsure,
    ConstraintsVoltageEnsure,
)
from ..components.cstr_generator import ConstraintGeneratorPowerRateMission

from ..components.sizing_diameter_scaling import SizingGeneratorDiameterScaling
from ..components.sizing_diameter import SizingGeneratorDiameter
from ..components.sizing_length_scaling import SizingGeneratorLengthScaling
from ..components.sizing_length import SizingGeneratorLength
from ..components.sizing_loss_coefficient_scaling import SizingGeneratorLossCoefficientScaling
from ..components.sizing_loss_coefficient import SizingGeneratorLossCoefficient
from ..components.sizing_resistance_scaling import SizingGeneratorPhaseResistanceScaling
from ..components.sizing_resistance import SizingGeneratorPhaseResistance
from ..components.sizing_torque_constant_scaling import SizingGeneratorTorqueConstantScaling
from ..components.sizing_torque_constant import SizingGeneratorTorqueConstant
from ..components.sizing_weight import SizingGeneratorWeight
from ..components.sizing_generator_cg_x import SizingGeneratorCGX

from ..components.perf_mission_rpm import PerformancesRPMMission
from ..components.perf_voltage_out_target import PerformancesVoltageOutTargetMission
from ..components.perf_current_rms_3_phases import PerformancesCurrentRMS3Phases
from ..components.perf_torque import PerformancesTorque
from ..components.perf_losses import PerformancesLosses
from ..components.perf_shaft_power_in import PerformancesShaftPowerIn
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_voltage_peak import PerformancesVoltagePeak
from ..components.perf_maximum import PerformancesMaximum

from ..components.sizing_generator import SizingGenerator

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_generator.xml"
NB_POINTS_TEST = 10


def test_diameter_scaling():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorDiameterScaling(generator_id="generator_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorDiameterScaling(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:scaling:diameter"
    ) == pytest.approx(1.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diameter():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorDiameter(generator_id="generator_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorDiameter(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:diameter", units="m"
    ) == pytest.approx(0.4824, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_length_scaling():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorLengthScaling(generator_id="generator_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorLengthScaling(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:scaling:length"
    ) == pytest.approx(0.403, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_length():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorLength(generator_id="generator_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorLength(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:length", units="m"
    ) == pytest.approx(0.03667, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_loss_coefficients_scaling():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorLossCoefficientScaling(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorLossCoefficientScaling(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:scaling:alpha"
    ) == pytest.approx(0.236, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:scaling:beta"
    ) == pytest.approx(340.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:scaling:gamma"
    ) == pytest.approx(152.60, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_loss_coefficients():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorLossCoefficient(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorLossCoefficient(generator_id="generator_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:generator:generator_1:loss_coefficient:alpha",
            units="W/N**2/m**2",
        )
        == pytest.approx(0.008, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:generator:generator_1:loss_coefficient:beta",
            units="W*s/rad",
        )
        == pytest.approx(26.56, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:generator:generator_1:loss_coefficient:gamma",
            units="W*s**2/rad**2",
        )
        == pytest.approx(0.0375, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_resistance_scaling():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorPhaseResistanceScaling(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorPhaseResistanceScaling(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:scaling:phase_resistance"
    ) == pytest.approx(1.40, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorPhaseResistance(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorPhaseResistance(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:phase_resistance", units="ohm"
    ) == pytest.approx(0.03206, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque_constant_scaling():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorTorqueConstantScaling(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorTorqueConstantScaling(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:scaling:torque_constant"
    ) == pytest.approx(2.43, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque_constant():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorTorqueConstant(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorTorqueConstant(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:torque_constant", units="N*m/A"
    ) == pytest.approx(4.617, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight():

    ivc = get_indep_var_comp(
        list_inputs(SizingGeneratorWeight(generator_id="generator_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGeneratorWeight(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:mass", units="kg"
    ) == pytest.approx(30.876, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_generator_cg_x():

    expected_cg = [2.69, 0.48, 1.99]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):

        ivc = get_indep_var_comp(
            list_inputs(SizingGeneratorCGX(generator_id="generator_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingGeneratorCGX(generator_id="generator_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:generator:generator_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_sizing():

    ivc = get_indep_var_comp(
        list_inputs(SizingGenerator(generator_id="generator_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingGenerator(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:mass", units="kg"
    ) == pytest.approx(30.876, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:CG:x", units="m"
    ) == pytest.approx(1.99, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:low_speed:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rpm_mission():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesRPMMission(generator_id="generator_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesRPMMission(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("rpm", units="min**-1") == pytest.approx(
        np.full(NB_POINTS_TEST, 2500.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_voltage_out_target():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesVoltageOutTargetMission(
                generator_id="generator_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesVoltageOutTargetMission(
            generator_id="generator_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("voltage_out_target", units="V") == pytest.approx(
        np.full(NB_POINTS_TEST, 400.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_rms_current_3_phases():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "ac_current_rms_out_one_phase", val=np.linspace(125, 145, NB_POINTS_TEST), units="A"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesCurrentRMS3Phases(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_current_rms_out", units="A") == pytest.approx(
        np.linspace(375.0, 435.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_apparent_power():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "ac_current_rms_out",
        val=np.linspace(375.0, 435.0, NB_POINTS_TEST),
        units="A",
    )
    ivc.add_output(
        "ac_voltage_rms_out",
        val=np.full(NB_POINTS_TEST, 400),
        units="V",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesApparentPower(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("apparent_power", units="kW") == pytest.approx(
        np.array([150.0, 152.7, 155.3, 158.0, 160.7, 163.3, 166.0, 168.7, 171.3, 174.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_active_power():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "apparent_power",
        val=np.array([150.0, 152.7, 155.3, 158.0, 160.7, 163.3, 166.0, 168.7, 171.3, 174.0]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesActivePower(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("active_power", units="kW") == pytest.approx(
        np.array([150.0, 152.7, 155.3, 158.0, 160.7, 163.3, 166.0, 168.7, 171.3, 174.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_shaft_power_in():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "active_power",
        units="kW",
        val=np.array([150.0, 152.7, 155.3, 158.0, 160.7, 163.3, 166.0, 168.7, 171.3, 174.0]),
    )
    ivc.add_output(
        "efficiency",
        val=np.array([0.953, 0.953, 0.952, 0.952, 0.952, 0.952, 0.952, 0.951, 0.951, 0.951]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesShaftPowerIn(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.array([157.4, 160.2, 163.1, 166.0, 168.8, 171.5, 174.4, 177.4, 180.1, 183.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_torque():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_in",
        val=np.array([157.4, 160.2, 163.1, 166.0, 168.8, 171.5, 174.4, 177.4, 180.1, 183.0]),
        units="kW",
    )
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500.0), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorque(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_in", units="N*m") == pytest.approx(
        np.array([601.2, 611.9, 623.0, 634.1, 644.8, 655.1, 666.2, 677.6, 687.9, 699.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_losses():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesLosses(generator_id="generator_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500.0), units="min**-1")
    ivc.add_output(
        "torque_in",
        val=np.array([601.2, 611.9, 623.0, 634.1, 644.8, 655.1, 666.2, 677.6, 687.9, 699.0]),
        units="N*m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesLosses(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("power_losses", units="kW") == pytest.approx(
        np.array([10.5, 10.8, 11.2, 11.5, 11.8, 12.2, 12.5, 12.9, 13.3, 13.7]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_efficiency():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_in",
        val=np.array([157.4, 160.2, 163.1, 166.0, 168.8, 171.5, 174.4, 177.4, 180.1, 183.0]),
        units="kW",
    )
    ivc.add_output(
        "power_losses",
        val=np.array([10.5, 10.8, 11.2, 11.5, 11.8, 12.2, 12.5, 12.9, 13.3, 13.7]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesEfficiency(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("efficiency") == pytest.approx(
        np.array([0.933, 0.933, 0.931, 0.931, 0.93, 0.929, 0.928, 0.927, 0.926, 0.925]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_voltage_peak():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "ac_voltage_rms_out",
        val=np.array(
            [199.47, 199.39, 199.31, 199.24, 199.42, 199.35, 199.28, 199.21, 199.14, 199.08]
        ),
        units="V",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesVoltagePeak(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_voltage_peak_out", units="V") == pytest.approx(
        np.array([244.3, 244.2, 244.1, 244.02, 244.24, 244.15, 244.07, 243.98, 243.9, 243.82]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()

    ivc.add_output(
        "ac_voltage_peak_out",
        val=np.array([244.3, 244.2, 244.1, 244.02, 244.24, 244.15, 244.07, 243.98, 243.9, 243.82]),
        units="V",
    )
    ivc.add_output(
        "ac_current_rms_out_one_phase", val=np.linspace(125, 145, NB_POINTS_TEST), units="A"
    )
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500.0), units="min**-1")
    ivc.add_output("torque_in", val=np.linspace(300.0, 348.0, NB_POINTS_TEST), units="N*m")
    ivc.add_output(
        "power_losses",
        val=np.array(
            [3700.3, 3781.0, 3863.2, 3946.7, 4031.7, 4118.1, 4205.9, 4295.2, 4385.8, 4477.9]
        ),
        units="W",
    )
    ivc.add_output(
        "shaft_power_in",
        val=np.array([78.5, 79.9, 81.3, 82.7, 84.1, 85.5, 86.9, 88.3, 89.7, 91.1]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:current_ac_max", units="A"
    ) == pytest.approx(145.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:voltage_ac_max", units="V"
    ) == pytest.approx(244.3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:torque_max", units="N*m"
    ) == pytest.approx(348.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:rpm_max", units="min**-1"
    ) == pytest.approx(2500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:losses_max", units="W"
    ) == pytest.approx(4477.9, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:shaft_power_max", units="W"
    ) == pytest.approx(91.1e3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_enforce_torque():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnforce(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )
    problem = run_system(ConstraintsTorqueEnforce(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:torque_rating", units="N*m"
    ) == pytest.approx(348, rel=1e-2)


def test_constraints_enforce_rpm():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRPMEnforce(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )
    problem = run_system(ConstraintsRPMEnforce(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:rpm_rating", units="min**-1"
    ) == pytest.approx(2500.0, rel=1e-2)


def test_constraints_voltage_enforce():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnforce(generator_id="generator_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnforce(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:voltage_caliber", units="V"
    ) == pytest.approx(400.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_torque():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnsure(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )
    problem = run_system(ConstraintsTorqueEnsure(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:generator:generator_1:torque_rating", units="N*m"
    ) == pytest.approx(-2.0, rel=1e-2)


def test_constraints_ensure_rpm():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRPMEnsure(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )
    problem = run_system(ConstraintsRPMEnsure(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:generator:generator_1:rpm_rating", units="min**-1"
    ) == pytest.approx(0.0, rel=1e-2)


def test_constraints_voltage_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnsure(generator_id="generator_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnsure(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:generator:generator_1:voltage_caliber", units="V"
    ) == pytest.approx(-0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintGeneratorPowerRateMission(generator_id="generator_1")),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintGeneratorPowerRateMission(generator_id="generator_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:shaft_power_rating", units="kW"
    ) == pytest.approx(91.1, rel=1e-2)

    problem.check_partials(compact_print=True)
