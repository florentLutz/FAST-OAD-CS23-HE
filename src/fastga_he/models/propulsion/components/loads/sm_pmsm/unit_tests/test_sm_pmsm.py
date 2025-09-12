# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth
import numpy as np
import pytest

import openmdao.api as om

from ..components.sizing_bore_diameter import SizingStatorBoreDiameter
from ..components.sizing_active_length import SizingActiveLength
from ..components.sizing_rotor_diameter import SizingRotorDiameter
from ..components.sizing_stator_yoke import SizingStatorYokeHeight
from ..components.sizing_slot_width import SizingSlotWidth
from ..components.sizing_ratio_x2p import SizingRatioX2p
from ..components.sizing_tooth_ratio import SizingToothRatio
from ..components.sizing_slot_height import SizingSlotHeight
from ..components.sizing_slot_section import SizingSlotSection
from ..components.sizing_conductor_section import SizingConductorSection
from ..components.sizing_conductor_length import SizingConductorLength
from ..components.sizing_conductors_number import SizingConductorsNumber
from ..components.sizing_winding_resistivity import SizingWindingResistivity
from ..components.sizing_external_stator_diameter import SizingExtStatorDiameter
from ..components.sizing_pmsm_cg_x import SizingPMSMCGX
from ..components.sizing_pmsm_cg_y import SizingPMSMCGY
from ..components.sizing_pmsm_drag import SizingPMSMDrag
from ..components.sizing_resistance import SizingResistance
from ..components.sizing_stator_core_weight import SizingStatorCoreWeight
from ..components.sizing_rotor_weight import SizingRotorWeight
from ..components.sizing_frame_weight import SizingFrameGeometry
from ..components.sizing_winding_stator_weight import SizingStatorWindingWeight
from ..components.sizing_pmsm_weight import SizingMotorWeight
from ..components.sizing_sm_pmsm import SizingSMPMSM

from ..components.perf_torque import PerformancesTorque
from ..components.perf_iron_losses import PerformancesIronLosses
from ..components.perf_joule_losses import PerformancesJouleLosses
from ..components.perf_windage_reynolds import PerformancesWindageReynolds
from ..components.perf_windage_friction_coeff import PerformancesWindageFrictionCoefficient
from ..components.perf_mechanical_losses import PerformancesMechanicalLosses
from ..components.perf_power_losses import PerformancesPowerLosses
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_electrical_frequency import PerformancesElectricalFrequency
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_current_rms import PerformancesCurrentRMS
from ..components.perf_current_rms_phase import PerformancesCurrentRMS1Phase
from ..components.perf_voltage_rms import PerformancesVoltageRMS
from ..components.perf_voltage_peak import PerformancesVoltagePeak
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_sm_pmsm import PerformancesSMPMSM

from ..components.pre_lca_prod_weight_per_fu import PreLCAMotorProdWeightPerFU

from ..components.lcc_sm_pmsm_cost import LCCSMPMSMCost
from ..components.lcc_sm_pmsm_operational_cost import LCCSMPMSMOperationalCost

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
from ..components.cstr_sm_pmsm import ConstraintPMSMPowerRateMission
from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_motor_new.xml"
NB_POINTS_TEST = 10


def test_bore_diameter():
    ivc = om.IndepVarComp()

    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:form_coefficient", val=0.6)
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:tangential_stress",
        val=50000,
        units="N/m**2",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:torque_rating",
        val=856.6,
        units="N*m",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingStatorBoreDiameter(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter", units="m"
    ) == pytest.approx(0.187, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rotor_diameter():
    ivc = om.IndepVarComp()

    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:radius_ratio", val=0.97)
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter",
        val=0.187,
        units="m",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingRotorDiameter(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_diameter", units="m"
    ) == pytest.approx(0.1814, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:airgap_thickness", units="m"
    ) == pytest.approx(0.0028, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_length():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter",
        val=0.187,
        units="m",
    )
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:form_coefficient", val=0.6)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingActiveLength(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:active_length", units="m"
    ) == pytest.approx(0.3117, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_yoke_height():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorYokeHeight(pmsm_id="motor_1")), __file__, XML_FILE
    )

    problem = run_system(SizingStatorYokeHeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_yoke_height", units="m"
    ) == pytest.approx(0.0351, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_tooth_ratio():
    ivc = get_indep_var_comp(list_inputs(SizingToothRatio(pmsm_id="motor_1")), __file__, XML_FILE)

    problem = run_system(SizingToothRatio(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:tooth_ratio"
    ) == pytest.approx(0.4407, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_slot_width():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:conductors_number", val=24)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:tooth_ratio", val=0.4407)
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter",
        val=0.187,
        units="m",
    )

    problem = run_system(SizingSlotWidth(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_width", units="m"
    ) == pytest.approx(0.0137, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_slot_height():
    ivc = get_indep_var_comp(list_inputs(SizingSlotHeight(pmsm_id="motor_1")), __file__, XML_FILE)

    problem = run_system(SizingSlotHeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_height"
    ) == pytest.approx(0.0358, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_slot_section():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_height", val=0.0358, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_width", val=0.0137, units="m"
    )

    problem = run_system(SizingSlotSection(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_section"
    ) == pytest.approx(0.00049046, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_conductor_section_area():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_section", val=0.00049046, units="m**2"
    )
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:slot_conductor_factor", val=1)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:slot_fill_factor", val=0.5)

    problem = run_system(SizingConductorSection(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:conductor_section"
    ) == pytest.approx(0.00024523, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_conductor_slot_number():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:number_of_phases", val=3)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:slots_per_poles_phases", val=2)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=2)

    problem = run_system(SizingConductorsNumber(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:conductors_number"
    ) == pytest.approx(24, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_conductor_length():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:cond_twisting_coeff", val=1.25)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:end_winding_coeff", val=1.4)

    problem = run_system(SizingConductorLength(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:conductor_length"
    ) == pytest.approx(0.545475, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_resistivity():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:winding_temperature", val=180, units="degC"
    )

    problem = run_system(SizingWindingResistivity(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:resistivity"
    ) == pytest.approx(2.736384e-08, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_x2p_ratio():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:radius_ratio", val=0.97)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=2)

    problem = run_system(SizingRatioX2p(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:ratiox2p"
    ) == pytest.approx(16.435, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_resistance():
    ivc = get_indep_var_comp(
        list_inputs(SizingResistance(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingResistance(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:resistance", units="ohm"
    ) == pytest.approx(
        0.0014608,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_external_stator_diameter():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter",
        val=0.187,
        units="m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_height", val=0.0358, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_yoke_height", val=0.0351, units="m"
    )

    problem = run_system(SizingExtStatorDiameter(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_diameter"
    ) == pytest.approx(0.3288, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_stator_core_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorCoreWeight(pmsm_id="motor_1")), __file__, XML_FILE
    )

    problem = run_system(SizingStatorCoreWeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_core_weight", units="kg"
    ) == pytest.approx(115.94, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_stator_winding_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorWindingWeight(pmsm_id="motor_1")), __file__, XML_FILE
    )

    problem = run_system(SizingStatorWindingWeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_winding_weight", units="kg"
    ) == pytest.approx(33.04, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rotor_weight():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_diameter", val=0.1814, units="m"
    )

    problem = run_system(SizingRotorWeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_weight", units="kg"
    ) == pytest.approx(56.97, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_frame_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingFrameGeometry(pmsm_id="motor_1")), __file__, XML_FILE
    )

    problem = run_system(SizingFrameGeometry(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:frame_weight", units="kg"
    ) == pytest.approx(19.52, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:frame_diameter", units="m"
    ) == pytest.approx(0.3564, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_SMPMSM_weight():
    ivc = get_indep_var_comp(list_inputs(SizingMotorWeight(pmsm_id="motor_1")), __file__, XML_FILE)

    problem = run_system(SizingMotorWeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mass", units="kg"
    ) == pytest.approx(225.59, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_motor_cg_x():
    expected_cg = [2.51, 0.25]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingPMSMCGX(pmsm_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingPMSMCGX(pmsm_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_cg_y():
    expected_cg = [1.5, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingPMSMCGY(pmsm_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingPMSMCGY(pmsm_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_drag():
    expected_drag_ls = [0.6, 0.0]
    expected_drag_cruise = [0.592, 0.0]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_drag_ls, expected_drag_cruise
    ):
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingPMSMDrag(pmsm_id="motor_1", position=option, low_speed_aero=ls_option)
                ),
                __file__,
                XML_FILE,
            )
            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingPMSMDrag(pmsm_id="motor_1", position=option, low_speed_aero=ls_option), ivc
            )

            if ls_option:
                assert problem.get_val(
                    "data:propulsion:he_power_train:SM_PMSM:motor_1:low_speed:CD0",
                ) * 1e3 == pytest.approx(ls_drag, rel=1e-2)
            else:
                assert problem.get_val(
                    "data:propulsion:he_power_train:SM_PMSM:motor_1:cruise:CD0",
                ) * 1e3 == pytest.approx(cruise_drag, rel=1e-2)

            # Slight error on reynolds is due to step
            problem.check_partials(compact_print=True)


def test_constraints_torque_enforce():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:torque_max", val=856.6, units="N*m"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnforce(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(856.63, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_rpm_enforce():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rpm_max", val=4500.0, units="min**-1"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRPMEnforce(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(4500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_enforce():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:voltage_ac_max", val=156.2, units="V"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnforce(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:voltage_caliber", units="V"
    ) == pytest.approx(156.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_torque_ensure():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:torque_max", val=856.6, units="N*m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:torque_rating", val=856.6, units="N*m"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnsure(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:SM_PMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_rpm_ensure():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rpm_max", val=4500.0, units="min**-1"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rpm_rating", val=5000.0, units="min**-1"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRPMEnsure(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:SM_PMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(-500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_ensure():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:voltage_ac_max", val=156.2, units="V"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:voltage_caliber", val=700.0, units="V"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnsure(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:SM_PMSM:motor_1:voltage_caliber", units="V"
    ) == pytest.approx(-543.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:shaft_power_max", val=1432.599, units="kW"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintPMSMPowerRateMission(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:shaft_power_rating", units="kW"
    ) == pytest.approx(1432.599, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque():
    ivc = om.IndepVarComp()

    ivc.add_output("shaft_power_out", np.linspace(30, 70, NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm", np.linspace(3500, 4500, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorque(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_out", units="N*m") == pytest.approx(
        [82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0], rel=1e-2
    )
    assert problem.get_val("shaft_power_for_power_rate", units="kW") == pytest.approx(
        np.linspace(30, 70, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_joule_losses():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:resistance", val=0.0014608, units="ohm"
    )

    ivc.add_output("ac_current_rms_in_one_phase", 1970.84 * np.ones(NB_POINTS_TEST), units="A")

    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:number_of_phases", val=3)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesJouleLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:joule_power_losses", units="W"
    ) == pytest.approx(17022 * np.ones(NB_POINTS_TEST), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_frequency():
    ivc = om.IndepVarComp()

    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=2)

    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesElectricalFrequency(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("electrical_frequency", units="s**-1") == pytest.approx(
        np.full(NB_POINTS_TEST, 532.33), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_iron_losses():
    ivc = om.IndepVarComp()

    ivc.add_output("electrical_frequency", np.full(NB_POINTS_TEST, 532.33), units="s**-1")
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:airgap_flux_density", val=0.9, units="T"
    )
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:mass", val=225.59, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesIronLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:iron_power_losses", units="W"
    ) == pytest.approx(np.full(NB_POINTS_TEST, 3625.25), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_windage_reynolds():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_diameter", val=0.1814, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:airgap_thickness", val=0.0028, units="m"
    )
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesWindageReynolds(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("airgap_reynolds_number") == pytest.approx(
        np.full(NB_POINTS_TEST, 27082.0), rel=1e-2
    )
    assert problem.get_val("rotor_end_reynolds_number") == pytest.approx(
        np.full(NB_POINTS_TEST, 877263.3), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_windage_friction_coefficient():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesWindageFrictionCoefficient(
                pmsm_id="motor_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output("airgap_reynolds_number", np.full(NB_POINTS_TEST, 27082.0))
    ivc.add_output("rotor_end_reynolds_number", np.full(NB_POINTS_TEST, 877263.3))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesWindageFrictionCoefficient(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("airgap_friction_coeff") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.001487), rel=1e-2
    )
    assert problem.get_val("rotor_end_friction_coeff") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.0094564), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_mechanical_losses():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesMechanicalLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output("airgap_friction_coeff", np.full(NB_POINTS_TEST, 0.001487))
    ivc.add_output("rotor_end_friction_coeff", np.full(NB_POINTS_TEST, 0.0094564))
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMechanicalLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mechanical_power_losses", units="W"
    ) == pytest.approx(1803.27 * np.ones(NB_POINTS_TEST), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_power_losses():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mechanical_power_losses",
        1803.27 * np.ones(NB_POINTS_TEST),
        units="W",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:iron_power_losses",
        3625.25 * np.ones(NB_POINTS_TEST),
        units="W",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:joule_power_losses",
        17022 * np.ones(NB_POINTS_TEST),
        units="W",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPowerLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("power_losses", units="W") == pytest.approx(
        22450.52 * np.ones(NB_POINTS_TEST), rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_efficiency():
    ivc = om.IndepVarComp()

    ivc.add_output("shaft_power_out", 1432.6 * np.ones(NB_POINTS_TEST), units="kW")
    ivc.add_output("power_losses", 22450.52 * np.ones(NB_POINTS_TEST), units="W")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEfficiency(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("efficiency") == pytest.approx(
        0.9846 * np.ones(NB_POINTS_TEST), rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_active_power():
    ivc = om.IndepVarComp()

    ivc.add_output("shaft_power_out", np.linspace(30, 70, NB_POINTS_TEST), units="kW")
    ivc.add_output(
        "efficiency",
        np.array([0.923, 0.928, 0.932, 0.936, 0.938, 0.941, 0.942, 0.944, 0.945, 0.946]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesActivePower(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("active_power", units="kW") == pytest.approx(
        [32.5, 37.1, 41.7, 46.3, 50.9, 55.5, 60.2, 64.7, 69.4, 74.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_apparent_power():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "active_power",
        np.array([32.5, 37.1, 41.7, 46.3, 50.9, 55.5, 60.2, 64.7, 69.4, 74.0]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesApparentPower(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("apparent_power", units="kW") == pytest.approx(
        [32.5, 37.1, 41.7, 46.3, 50.9, 55.5, 60.2, 64.7, 69.4, 74.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_rms_current():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "torque_out",
        np.array([82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0]),
        units="N*m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:torque_constant", 0.4312, units="N*m/A"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCurrentRMS(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("ac_current_rms_in", units="A") == pytest.approx(
        [190.17, 211.04, 231.91, 250.46, 269.02, 285.25, 301.48, 315.4, 331.63, 345.55],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_rms_current_1_phase():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "ac_current_rms_in",
        np.array([56.1, 62.3, 68.4, 73.9, 79.4, 84.2, 89.0, 93.1, 97.9, 102.0]),
        units="A",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesCurrentRMS1Phase(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_current_rms_in_one_phase", units="A") == pytest.approx(
        [18.7, 20.8, 22.8, 24.6, 26.5, 28.1, 29.7, 31.0, 32.6, 34.0],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_rms_voltage():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "apparent_power",
        np.array([32.5, 37.1, 41.7, 46.3, 50.9, 55.5, 60.2, 64.7, 69.4, 74.0]),
        units="kW",
    )
    ivc.add_output(
        "ac_current_rms_in",
        np.array([106.5, 118.2, 129.9, 140.3, 150.6, 159.7, 168.8, 176.6, 185.7, 193.5]),
        units="A",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesVoltageRMS(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_voltage_rms_in", units="V") == pytest.approx(
        [305.1, 313.8, 321.0, 330.0, 337.9, 347.5, 356.6, 366.3, 373.7, 382.4],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_peak_voltage():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "ac_voltage_rms_in",
        np.array([101.7, 104.6, 107.0, 110.0, 112.7, 115.8, 118.9, 122.1, 124.6, 127.5]),
        units="V",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesVoltagePeak(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_voltage_peak_in", units="V") == pytest.approx(
        [124.6, 128.1, 131.0, 134.7, 138.0, 141.8, 145.6, 149.5, 152.6, 156.2], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "torque_out",
        np.array([82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0]),
        units="N*m",
    )
    ivc.add_output("rpm", np.linspace(3500, 4500, NB_POINTS_TEST), units="min**-1")
    ivc.add_output(
        "ac_voltage_peak_in",
        units="V",
        val=np.array([124.6, 128.1, 131.0, 134.7, 138.0, 141.8, 145.6, 149.5, 152.6, 156.2]),
    )
    ivc.add_output(
        "ac_current_rms_in_one_phase",
        np.array([56.1, 62.3, 68.4, 73.9, 79.4, 84.2, 89.0, 93.1, 97.9, 102.0]),
        units="A",
    )
    ivc.add_output(
        "power_losses",
        np.array([2.52, 2.66, 2.82, 2.98, 3.14, 3.3, 3.47, 3.63, 3.81, 3.98]),
        units="kW",
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, 10), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:current_ac_max", units="A"
    ) == pytest.approx(102.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:voltage_ac_max", units="V"
    ) == pytest.approx(156.2, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:torque_max", units="N*m"
    ) == pytest.approx(149.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rpm_max", units="min**-1"
    ) == pytest.approx(4500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:losses_max", units="kW"
    ) == pytest.approx(3.98, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:shaft_power_max", units="kW"
    ) == pytest.approx(70.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performance_SM_PMSM():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesSMPMSM(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )

    ivc.add_output("shaft_power_out", 1432.6 * np.ones(NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSMPMSM(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("ac_current_rms_in", units="A") == pytest.approx(
        np.full(NB_POINTS_TEST, 1970.8434), rel=1e-2
    )
    assert problem.get_val("ac_current_rms_in_one_phase", units="A") == pytest.approx(
        np.full(NB_POINTS_TEST, 656.9478), rel=1e-2
    )
    assert problem.get_val("ac_voltage_rms_in", units="V") == pytest.approx(
        np.full(NB_POINTS_TEST, 727), rel=1e-2
    )
    assert problem.get_val("ac_voltage_peak_in", units="V") == pytest.approx(
        np.full(NB_POINTS_TEST, 890.4), rel=1e-2
    )

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))

    problem.check_partials(compact_print=True)


def test_sizing_SM_PMSM():
    ivc = get_indep_var_comp(list_inputs(SizingSMPMSM(pmsm_id="motor_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingSMPMSM(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter", units="m"
    ) == pytest.approx(0.187, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_diameter", units="m"
    ) == pytest.approx(0.1122, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:active_length", units="m"
    ) == pytest.approx(0.3117, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_yoke_height", units="m"
    ) == pytest.approx(0.0351, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_width"
    ) == pytest.approx(0.0137, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_height", units="m"
    ) == pytest.approx(0.0358, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_diameter", units="m"
    ) == pytest.approx(0.3288, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_core_weight", units="kg"
    ) == pytest.approx(115.94, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_winding_weight", units="kg"
    ) == pytest.approx(33.04, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_weight", units="kg"
    ) == pytest.approx(21.798, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:frame_weight", units="kg"
    ) == pytest.approx(19.52, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:frame_diameter", units="m"
    ) == pytest.approx(0.3564, rel=1e-2)
    problem.check_partials(compact_print=True)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mass", units="kg"
    ) == pytest.approx(190.46, rel=1e-2)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mass",
        "data:environmental_impact:aircraft_per_fu",
        "data:TLAR:aircraft_lifespan",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCAMotorProdWeightPerFU(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mass_per_fu", units="kg"
    ) == pytest.approx(4.512e-4, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cost():
    ivc = get_indep_var_comp(list_inputs(LCCSMPMSMCost(pmsm_id="motor_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCSMPMSMCost(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:purchase_cost", units="USD"
    ) == pytest.approx(24521.15, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_operational_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:purchase_cost",
        24521.15,
        units="USD",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCSMPMSMOperationalCost(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:operational_cost", units="USD/yr"
    ) == pytest.approx(1634.74, rel=1e-2)

    problem.check_partials(compact_print=True)
