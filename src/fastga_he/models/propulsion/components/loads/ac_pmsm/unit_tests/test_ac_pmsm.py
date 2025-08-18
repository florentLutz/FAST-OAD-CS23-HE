# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.sizing_diameter import SizingStatorDiameter
from ..components.sizing_active_length import SizingActiveLength
from ..components.sizing_rotor_diameter import SizingRotorDiameter
from ..components.sizing_stator_yoke import SizingStatorYokeHeight
from ..components.sizing_slot_width import SizingSlotWidth

# from ..components.sizing_slot_height_new import SizingSlotHeightNew
from ..components.sizing_slot_section import SizingSlotSection
from ..components.sizing_conductor_section import SizingConductorSection
from ..components.sizing_conductor_length import SizingConductorLength
from ..components.sizing_conductors_number import SizingConductorsNumber
from ..components.sizing_winding_resistivity import SizingWindingResistivity
from ..components.sizing_external_stator_diameter import SizingExtStatorDiameter

# from ..components.sizing_resistance_new import SizingResistanceNew
from ..components.sizing_resistance_new2 import SizingResistanceNew2
from ..components.sizing_stator_core_weight import SizingStatorCoreWeight
from ..components.sizing_rotor_weight import SizingRotorWeight
from ..components.sizing_frame_weight import SizingFrameWeight
from ..components.sizing_winding_stator_weight import SizingStatorWindingWeight
from ..components.sizing_pmsm_weight import SizingMotorWeight
from ..components.perf_torque import PerformancesTorque
from ..components.perf_iron_losses import PerformancesIronLosses
from ..components.perf_Joule_losses import PerformancesJouleLosses
from ..components.perf_Joule_losses_2 import PerformancesJouleLosses2
from ..components.perf_mechanical_losses import PerformancesMechanicalLosses
from ..components.perf_power_losses import PerformancesPowerLosses
from ..components.perf_efficiency import PerformancesEfficiency

# from ..components.specific_power import perfSpecificPower
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower

# from ..components.perf_current_rms import PerformancesCurrentRMS
from ..components.perf_current_rms_phase import PerformancesCurrentRMS1Phase
from ..components.perf_voltage_rms import PerformancesVoltageRMS
from ..components.perf_voltage_peak import PerformancesVoltagePeak
from ..components.perf_maximum import PerformancesMaximum
from ..components.pre_lca_prod_weight_per_fu import PreLCAMotorProdWeightPerFU
from ..components.sizing_ac_pmsm import SizingACPMSM
from ..components.sizing_ac_pmsm_old import SizingACPMSMNEW
from ..components.perf_ac_pmsm import PerformancesACPMSM
from ..components.perf_ac_pmsm_old import PerformancesACPMSMNEW


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
from ..components.cstr_pmsm import ConstraintPMSMPowerRateMission

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_motor_new.xml"
NB_POINTS_TEST = 10


def test_diameter():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorDiameter(pmsm_id="motor_1")), __file__, XML_FILE
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:Form_coefficient", val=0.6)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Tangential_stress", val=50000, units="N/m**2"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingStatorDiameter(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:diameter", units="m"
    ) == pytest.approx(0.187, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rot_diameter():
    ivc = get_indep_var_comp(
        list_inputs(SizingRotorDiameter(pmsm_id="motor_1")), __file__, XML_FILE
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:radius_ratio", val=0.97)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingRotorDiameter(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rot_diameter", units="m"
    ) == pytest.approx(0.1814, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_length():
    ivc = get_indep_var_comp(list_inputs(SizingActiveLength(pmsm_id="motor_1")), __file__, XML_FILE)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:Form_coefficient", val=0.6)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingActiveLength(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", units="m"
    ) == pytest.approx(0.3117, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_yoke_height():
    ivc = get_indep_var_comp(list_inputs(SizingActiveLength(pmsm_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:surface_current_density",
        val=111.100,
        units="A/m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:airgap_flux_density", val=0.9, units="T"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:yoke_flux_density", val=1.2, units="T"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:radius_ratio", val=0.6)

    problem = run_system(SizingStatorYokeHeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_yoke_height"
    ) == pytest.approx(0.0351, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_slot_width():
    ivc = get_indep_var_comp(list_inputs(SizingSlotWidth(pmsm_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:conductors_number", val=24)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:surface_current_density",
        val=111.100,
        units="A/m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:airgap_flux_density", val=0.9, units="T"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:tooth_flux_density", val=1.3, units="T"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:radius_ratio", val=0.97)

    problem = run_system(SizingSlotWidth(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width"
    ) == pytest.approx(0.0137, rel=1e-2)

    problem.check_partials(compact_print=True)


"""def test_slot_height():
    ivc = get_indep_var_comp(list_inputs(SizingSlotHeight(pmsm_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:density_current_ac_max', val=8.1e6, units="A/m**2")
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:winding_factor', val=0.97)
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number', val=2)
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:slot_conductor_factor', val=1)
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:tooth_flux_density', val=1.3, units="T")
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:surface_current_density', val=111.100, units="A/m")
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:slot_fill_factor', val=0.5)
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:Tangential_stress', val=50000, units="N/m**2")
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:airgap_flux_density', val=0.9, units="T")
    ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:radius_ratio', val=0.6)

    problem = run_system(SizingSlotHeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height"
    ) == pytest.approx(0.0358, rel=1e-2)

    problem.check_partials(compact_print=True)"""


# def test_slot_height_new():
#     ivc = get_indep_var_comp(
#         list_inputs(SizingSlotHeightNew(pmsm_id="motor_1")), __file__, XML_FILE
#     )
#     # Run problem and check obtained value(s) is/(are) correct
#     ivc.add_output(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:density_current_ac_max",
#         val=8.1e6,
#         units="A/m**2",
#     )
#     ivc.add_output(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:current_ac_max", val=1986, units="A"
#     )
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_conductor_factor", val=1)
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_fill_factor", val=0.5)
#     ivc.add_output(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width", val=0.0137, units="m"
#     )
#
#     problem = run_system(SizingSlotHeightNew(pmsm_id="motor_1"), ivc)
#
#     assert problem.get_val(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height"
#     ) == pytest.approx(0.0358, rel=1e-3)
#
#     problem.check_partials(compact_print=True)


def test_slot_section():
    ivc = get_indep_var_comp(list_inputs(SizingSlotSection(pmsm_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height", val=0.0358, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width", val=0.0137, units="m"
    )

    problem = run_system(SizingSlotSection(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_section"
    ) == pytest.approx(0.00049046, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cond_section():
    ivc = get_indep_var_comp(
        list_inputs(SizingConductorSection(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_section", val=0.00049046, units="m**2"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_conductor_factor", val=1)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_fill_factor", val=0.5)

    problem = run_system(SizingConductorSection(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:conductor_section"
    ) == pytest.approx(0.00024523, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cond_number():
    ivc = get_indep_var_comp(
        list_inputs(SizingConductorsNumber(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:number_of_phases", val=3)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slots_per_poles_phases", val=2)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)

    problem = run_system(SizingConductorsNumber(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:conductors_number"
    ) == pytest.approx(24, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cond_length():
    ivc = get_indep_var_comp(
        list_inputs(SizingConductorLength(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:cond_twisting_coeff", val=1.25)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)

    problem = run_system(SizingConductorLength(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:conductor_length"
    ) == pytest.approx(0.545475, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_resistivity():
    ivc = get_indep_var_comp(
        list_inputs(SizingWindingResistivity(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:winding_temperature", val=180, units="degC"
    )

    problem = run_system(SizingWindingResistivity(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:resistivity"
    ) == pytest.approx(2.736384e-08, rel=1e-3)

    problem.check_partials(compact_print=True)


# def test_Resistance():
#     ivc = get_indep_var_comp(
#         list_inputs(SizingResistanceNew(pmsm_id="motor_1")), __file__, XML_FILE
#     )
#
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:number_of_phases", val=3)
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slots_per_poles_phases", val=2)
#     ivc.add_output(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
#     )
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_fill_factor", val=0.5)
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:cond_twisting_coeff", val=1.25)
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)
#     ivc.add_output(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height", val=0.0358, units="m"
#     )
#     ivc.add_output(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width", val=0.0137, units="m"
#     )
#     ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_conductor_factor", val=1)
#     ivc.add_output(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:winding_temperature", val=180, units="degC"
#     )
#
#     # Run problem and check obtained value(s) is/(are) correct
#     problem = run_system(
#         SizingResistanceNew(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
#     )
#
#     assert problem.get_val(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:resistance", units="ohm"
#     ) == pytest.approx(0.0014608, rel=1e-2)
#
#     problem.check_partials(compact_print=True)


def test_Resistance2():
    ivc = get_indep_var_comp(
        list_inputs(SizingResistanceNew2(pmsm_id="motor_1")), __file__, XML_FILE
    )

    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:conductors_number", val=24)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:conductor_length", val=0.545475, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:conductor_section",
        val=0.00024523,
        units="m**2",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:resistivity", val=2.736384e-08, units="ohm*m"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingResistanceNew2(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:resistance", units="ohm"
    ) == pytest.approx(
        0.0014608,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_Ext_Diameter():
    ivc = get_indep_var_comp(
        list_inputs(SizingExtStatorDiameter(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height", val=0.0358, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_yoke_height", val=0.0351, units="m"
    )
    problem = run_system(SizingExtStatorDiameter(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:ext_stator_diameter"
    ) == pytest.approx(0.3288, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_stator_core_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorCoreWeight(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:number_of_phases", val=3)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slots_per_poles_phases", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:ext_stator_diameter", val=0.3288, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width", val=0.01348, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height", val=0.0358, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:magn_mat_density", val=8150, units="kg/m**3"
    )
    problem = run_system(SizingStatorCoreWeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_core_weight", units="kg"
    ) == pytest.approx(115.94, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_stator_winding_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorWindingWeight(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:number_of_phases", val=3)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slots_per_poles_phases", val=2)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_fill_factor", val=0.5)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:cond_twisting_coeff", val=1.25)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height", val=0.0358, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width", val=0.01348, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:cond_mat_density", val=8960, units="kg/m**3"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:insul_mat_density", val=1400, units="kg/m**3"
    )

    problem = run_system(SizingStatorWindingWeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_winding_weight", units="kg"
    ) == pytest.approx(33.04, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rotor_weight():
    ivc = get_indep_var_comp(list_inputs(SizingRotorWeight(pmsm_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rot_diameter", val=0.1814, units="m"
    )

    problem = run_system(SizingRotorWeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rotor_weight", units="kg"
    ) == pytest.approx(56.97, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_frame_weight():
    ivc = get_indep_var_comp(list_inputs(SizingFrameWeight(pmsm_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:ext_stator_diameter", val=0.3288, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:frame_density", val=2100, units="kg/m**3"
    )

    problem = run_system(SizingFrameWeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:frame_weight", units="kg"
    ) == pytest.approx(19.52, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:frame_diameter", units="m"
    ) == pytest.approx(0.3564, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_ACPMSM_weight():
    ivc = get_indep_var_comp(list_inputs(SizingMotorWeight(pmsm_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_core_weight", val=116.40, units="kg"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_winding_weight", val=32.69, units="kg"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rotor_weight", val=56.97, units="kg"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:frame_weight", val=19.52, units="kg"
    )

    problem = run_system(SizingMotorWeight(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:mass", units="kg"
    ) == pytest.approx(225.59, rel=1e-2)

    problem.check_partials(compact_print=True)


"""
def test_motor_cg_x():
    expected_cg = [2.39, 0.25]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingACPMSMCGX(pmsm_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingACPMSMCGX(pmsm_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:ACPMSM:motor_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_cg_y():
    expected_cg = [1.5, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingACPMSMCGY(pmsm_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingACPMSMCGY(pmsm_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:ACPMSM:motor_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_drag():
    expected_drag_ls = [0.357, 0.0]
    expected_drag_cruise = [0.352, 0.0]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_drag_ls, expected_drag_cruise
    ):
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingACPMSMDrag(pmsm_id="motor_1", position=option, low_speed_aero=ls_option)
                ),
                __file__,
                XML_FILE,
            )
            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingACPMSMDrag(pmsm_id="motor_1", position=option, low_speed_aero=ls_option), ivc
            )

            if ls_option:
                assert problem.get_val(
                    "data:propulsion:he_power_train:ACPMSM:motor_1:low_speed:CD0",
                ) * 1e3 == pytest.approx(ls_drag, rel=1e-2)
            else:
                assert problem.get_val(
                    "data:propulsion:he_power_train:ACPMSM:motor_1:cruise:CD0",
                ) * 1e3 == pytest.approx(cruise_drag, rel=1e-2)

            # Slight error on reynolds is due to step
            problem.check_partials(compact_print=True)
"""


def test_constraints_torque_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnforce(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnforce(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(856.63, rel=1e-2)

    problem.check_partials(compact_print=True)


"""def test_constraints_rpm_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRPMEnforce(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRPMEnforce(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(4500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnforce(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnforce(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:voltage_caliber", units="V"
    ) == pytest.approx(156.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_torque_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnsure(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnsure(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:ACPMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_rpm_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRPMEnsure(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRPMEnsure(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:ACPMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(-500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnsure(pmsm_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnsure(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:ACPMSM:motor_1:voltage_caliber", units="V"
    ) == pytest.approx(-543.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintACPMSMPowerRateMission(pmsm_id="motor_1")),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintACPMSMPowerRateMission(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:shaft_power_rating", units="kW"
    ) == pytest.approx(70.0, rel=1e-2)

    problem.check_partials(compact_print=True)
"""


def test_torque():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(PerformancesTorque(number_of_points=NB_POINTS_TEST)), __file__, XML_FILE
    )
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


def test_Joulelosses():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesJouleLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )

    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:number_of_phases", val=3)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slots_per_poles_phases", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_fill_factor", val=0.5)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:cond_twisting_coeff", val=1.25)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height", val=0.0358, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width", val=0.0137, units="m"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_conductor_factor", val=1)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:winding_temperature", val=180, units="degC"
    )

    ivc.add_output("ac_current_rms_in_one_phase", 1970.84 * np.ones(NB_POINTS_TEST), units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesJouleLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Joule_power_losses", units="W"
    ) == pytest.approx(5783.96 * np.ones(NB_POINTS_TEST), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_Joulelosses2():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesJouleLosses2(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:resistance", val=0.0014608, units="ohm"
    )

    ivc.add_output("ac_current_rms_in_one_phase", 1970.84 * np.ones(NB_POINTS_TEST), units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesJouleLosses2(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Joule_power_losses", units="W"
    ) == pytest.approx(5674 * np.ones(NB_POINTS_TEST), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_Ironlosses():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesIronLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )

    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:airgap_flux_density", val=0.9, units="T"
    )
    # ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:mass', val=225.59, units="kg")
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesIronLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:iron_power_losses", units="W"
    ) == pytest.approx(3625.25 * np.ones(NB_POINTS_TEST), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_Mechanicallosses():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesMechanicalLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rot_diameter", val=0.1814, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Airgap_thickness", val=0.0028, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:airgap_flux_density", val=0.9, units="T"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rotor_weight", val=56.97, units="kg"
    )
    # ivc.add_output('data:propulsion:he_power_train:ACPMSM:motor_1:mass', val=225.59, units="kg")
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMechanicalLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:mechanical_power_losses", units="W"
    ) == pytest.approx(1803.27 * np.ones(NB_POINTS_TEST), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_Power_losses():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesPowerLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:mechanical_power_losses",
        1803.27 * np.ones(NB_POINTS_TEST),
        units="W",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:iron_power_losses",
        3625.25 * np.ones(NB_POINTS_TEST),
        units="W",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Joule_power_losses",
        5783.96 * np.ones(NB_POINTS_TEST),
        units="W",
    )
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPowerLosses(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("power_losses", units="W") == pytest.approx(
        22424.96 * np.ones(NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_efficiency():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesEfficiency(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", 1432.6 * np.ones(NB_POINTS_TEST), units="kW")
    ivc.add_output("power_losses", 22424.96 * np.ones(NB_POINTS_TEST), units="W")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEfficiency(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("efficiency") == pytest.approx(
        0.9846 * np.ones(NB_POINTS_TEST), rel=1e-3
    )

    problem.check_partials(compact_print=True)


# def test_specific_power():
#     ivc = get_indep_var_comp(
#         list_inputs(perfSpecificPower(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
#         __file__,
#         XML_FILE,
#     )
#     ivc.add_output("shaft_power_out", 1432.6 * np.ones(NB_POINTS_TEST), units="kW")
#     ivc.add_output(
#         "data:propulsion:he_power_train:ACPMSM:motor_1:mechanical_power_losses",
#         1803.27 * np.ones(NB_POINTS_TEST),
#         units="W",
#     )
#
#     # Run problem and check obtained value(s) is/(are) correct
#     problem = run_system(perfSpecificPower(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc)
#
#     assert problem.get_val("specific_power", units="W/kg") == pytest.approx(
#         6300 * np.ones(NB_POINTS_TEST), rel=1e-2
#     )
#
#     problem.check_partials(compact_print=True)


def test_active_power():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesActivePower(number_of_points=NB_POINTS_TEST)), __file__, XML_FILE
    )
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
    ivc = get_indep_var_comp(
        list_inputs(PerformancesApparentPower(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
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


# def test_rms_current():
#     ivc = get_indep_var_comp(
#         list_inputs(PerformancesCurrentRMS(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
#         __file__,
#         XML_FILE,
#     )
#     ivc.add_output(
#         "torque_out",
#         np.array([82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0]),
#         units="N*m",
#     )
#
#     # Run problem and check obtained value(s) is/(are) correct
#     problem = run_system(
#         PerformancesCurrentRMS(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
#     )
#
#     assert problem.get_val("ac_current_rms_in", units="A") == pytest.approx(
#         [56.1, 62.3, 68.4, 73.9, 79.4, 84.2, 89.0, 93.1, 97.9, 102.0],
#         rel=1e-2,
#     )
#
#     problem.check_partials(compact_print=True)


def test_rms_current_1_phase():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesCurrentRMS1Phase(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
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
    ivc = get_indep_var_comp(
        list_inputs(PerformancesVoltageRMS(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
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
    ivc = get_indep_var_comp(
        list_inputs(PerformancesVoltagePeak(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
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
        "data:propulsion:he_power_train:ACPMSM:motor_1:current_ac_max", units="A"
    ) == pytest.approx(102.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:voltage_ac_max", units="V"
    ) == pytest.approx(156.2, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:torque_max", units="N*m"
    ) == pytest.approx(149.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rpm_max", units="min**-1"
    ) == pytest.approx(4500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:losses_max", units="kW"
    ) == pytest.approx(3.98, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:shaft_power_max", units="kW"
    ) == pytest.approx(70.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_ACPMSM():
    ivc = get_indep_var_comp(list_inputs(SizingACPMSM(pmsm_id="motor_1")), __file__, XML_FILE)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:Form_coefficient", val=0.6)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Tangential_stress", val=50000, units="N/m**2"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:radius_ratio", val=0.97)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:surface_current_density",
        val=111.100,
        units="A/m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:airgap_flux_density", val=0.9, units="T"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:yoke_flux_density", val=1.2, units="T"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:number_of_phases", val=3)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slots_per_poles_phases", val=2)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_fill_factor", val=0.5)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:cond_twisting_coeff", val=1.25)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:frame_density", val=2100, units="kg/m**3"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:current_ac_max", val=1021, units="A"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:magn_mat_density", val=8150, units="kg/m**3"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:cond_mat_density", val=8960, units="kg/m**3"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:insul_mat_density", val=1400, units="kg/m**3"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:density_current_ac_max",
        val=8.1e6,
        units="A/m**2",
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:winding_factor", val=0.97)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_conductor_factor", val=1)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:tooth_flux_density", val=1.3, units="T"
    )
    # Run problem and check obtained value(s) is/(are) correct

    problem = run_system(SizingACPMSM(pmsm_id="motor_1"), ivc)

    # om.n2(problem)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:diameter", units="m"
    ) == pytest.approx(0.1494, rel=1e-2)
    #
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rot_diameter", units="m"
    ) == pytest.approx(0.1449, rel=1e-2)
    #
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", units="m"
    ) == pytest.approx(0.2491, rel=1e-2)
    #
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_yoke_height", units="m"
    ) == pytest.approx(0.0280, rel=1e-2)
    #
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width"
    ) == pytest.approx(0.0109, rel=1e-2)
    #
    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height", units="m"
    ) == pytest.approx(0.0358, rel=1e-2)
    #
    # assert problem.get_val(
    #     "data:propulsion:he_power_train:ACPMSM:motor_1:ext_stator_diameter", units="m"
    # ) == pytest.approx(0.277, rel=1e-2)
    #
    # assert problem.get_val(
    #     "data:propulsion:he_power_train:ACPMSM:motor_1:stator_core_weight", units="kg"
    # ) == pytest.approx(115.94, rel=1e-2)
    #
    # assert problem.get_val(
    #     "data:propulsion:he_power_train:ACPMSM:motor_1:stator_winding_weight", units="kg"
    # ) == pytest.approx(33.04, rel=1e-2)
    #
    # assert problem.get_val(
    #     "data:propulsion:he_power_train:ACPMSM:motor_1:rotor_weight", units="kg"
    # ) == pytest.approx(56.97, rel=1e-2)
    #
    # assert problem.get_val(
    #     "data:propulsion:he_power_train:ACPMSM:motor_1:frame_weight", units="kg"
    # ) == pytest.approx(19.52, rel=1e-2)
    #
    # assert problem.get_val(
    #     "data:propulsion:he_power_train:ACPMSM:motor_1:frame_diameter", units="m"
    # ) == pytest.approx(0.3564, rel=1e-2)
    # problem.check_partials(compact_print=True)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:mass", units="kg"
    ) == pytest.approx(225.59, rel=1e-2)


def test_sizing_ACPMSM_new():
    ivc = get_indep_var_comp(list_inputs(SizingACPMSMNEW(pmsm_id="motor_1")), __file__, XML_FILE)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:Form_coefficient", val=0.6)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Tangential_stress", val=50000, units="N/m**2"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:radius_ratio", val=0.97)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:surface_current_density",
        val=111.100,
        units="A/m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:airgap_flux_density", val=0.9, units="T"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:yoke_flux_density", val=1.2, units="T"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:number_of_phases", val=3)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slots_per_poles_phases", val=2)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_fill_factor", val=0.5)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:cond_twisting_coeff", val=1.25)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:frame_density", val=2100, units="kg/m**3"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:current_ac_max", val=1986, units="A"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:magn_mat_density", val=8150, units="kg/m**3"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:cond_mat_density", val=8960, units="kg/m**3"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:insul_mat_density", val=1400, units="kg/m**3"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:density_current_ac_max",
        val=8.1e6,
        units="A/m**2",
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:winding_factor", val=0.97)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_conductor_factor", val=1)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:tooth_flux_density", val=1.3, units="T"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:winding_temperature", val=180, units="degC"
    )
    # Run problem and check obtained value(s) is/(are) correct

    problem = run_system(SizingACPMSMNEW(pmsm_id="motor_1"), ivc)

    # om.n2(problem)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:diameter", units="m"
    ) == pytest.approx(0.187, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rot_diameter", units="m"
    ) == pytest.approx(0.1814, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", units="m"
    ) == pytest.approx(0.3117, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_yoke_height", units="m"
    ) == pytest.approx(0.0351, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width"
    ) == pytest.approx(0.0137, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height", units="m"
    ) == pytest.approx(0.0358, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:ext_stator_diameter", units="m"
    ) == pytest.approx(0.3288, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_core_weight", units="kg"
    ) == pytest.approx(115.94, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:stator_winding_weight", units="kg"
    ) == pytest.approx(33.04, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rotor_weight", units="kg"
    ) == pytest.approx(56.97, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:frame_weight", units="kg"
    ) == pytest.approx(19.52, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:frame_diameter", units="m"
    ) == pytest.approx(0.3564, rel=1e-2)
    problem.check_partials(compact_print=True)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:mass", units="kg"
    ) == pytest.approx(225.59, rel=1e-2)


def test_performance_ACPMSM():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesACPMSM(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", 1432.6 * np.ones(NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:airgap_flux_density", val=0.9, units="T"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:number_of_phases", val=3)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slots_per_poles_phases", val=2)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:cond_twisting_coeff", val=1.25)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_fill_factor", val=0.5)
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:slot_conductor_factor", val=1)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:winding_temperature", val=180, units="degC"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_height", val=0.0358, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:slot_width", val=0.01348, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rot_diameter", val=0.1814, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Airgap_thickness", val=0.0028, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rotor_weight", val=56.97, units="kg"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesACPMSM(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    om.n2(problem)

    assert problem.get_val("ac_current_rms_in", units="A") == pytest.approx(
        1970.8434 * np.ones(NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val("ac_current_rms_in_one_phase", units="A") == pytest.approx(
        1970.8434 * np.ones(NB_POINTS_TEST) / 3, rel=1e-2
    )
    assert problem.get_val("ac_voltage_rms_in", units="V") == pytest.approx(
        727 * np.ones(NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val("ac_voltage_peak_in", units="V") == pytest.approx(
        727 * np.ones(NB_POINTS_TEST) * np.sqrt(3 / 2), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_performance_ACPMSM2():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesACPMSMNEW(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", 1432.6 * np.ones(NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:number_of_phases", val=3)

    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:airgap_flux_density", val=0.9, units="T"
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:end_winding_coeff", val=1.4)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:active_length", val=0.3117, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rot_diameter", val=0.1814, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Airgap_thickness", val=0.0028, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:rotor_weight", val=56.97, units="kg"
    )

    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:resistance", val=0.0014608, units="ohm"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesACPMSMNEW(pmsm_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    om.n2(problem)

    assert problem.get_val("ac_current_rms_in", units="A") == pytest.approx(
        1970.8434 * np.ones(NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val("ac_current_rms_in_one_phase", units="A") == pytest.approx(
        1970.8434 * np.ones(NB_POINTS_TEST) / 3, rel=1e-2
    )
    assert problem.get_val("ac_voltage_rms_in", units="V") == pytest.approx(
        727 * np.ones(NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val("ac_voltage_peak_in", units="V") == pytest.approx(
        727 * np.ones(NB_POINTS_TEST) * np.sqrt(3 / 2), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:ACPMSM:motor_1:mass",
        "data:environmental_impact:aircraft_per_fu",
        "data:TLAR:aircraft_lifespan",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCAMotorProdWeightPerFU(pmsm_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ACPMSM:motor_1:mass_per_fu", units="kg"
    ) == pytest.approx(3.238e-5, rel=1e-3)

    problem.check_partials(compact_print=True)
