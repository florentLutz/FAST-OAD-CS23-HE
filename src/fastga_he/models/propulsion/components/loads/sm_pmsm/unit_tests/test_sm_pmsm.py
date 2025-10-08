# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth
import numpy as np
import pytest

import openmdao.api as om

from ..components.sizing_bore_diameter import SizingStatorBoreDiameter
from ..components.sizing_active_length import SizingActiveLength
from ..components.sizing_frame_dimension import SizingFrameDimension
from ..components.sizing_rotor_diameter import SizingRotorDiameter
from ..components.sizing_stator_yoke_height import SizingStatorYokeHeight
from ..components.sizing_slot_width import SizingSlotWidth
from ..components.sizing_ratio_x2p import SizingRatioX2p
from ..components.sizing_slot_height import SizingSlotHeight
from ..components.sizing_radius_ratio import SizingRadiusRatio
from ..components.sizing_slot_section_area import SizingSlotSectionArea
from ..components.sizing_conductor_section_area_per_slot import SizingConductorSectionAreaPerSlot
from ..components.sizing_single_conductor_cable_length import SizingSingleConductorCableLength
from ..components.sizing_conductor_slot_number import SizingConductorSlotNumber
from ..components.sizing_pouillet_geometry_factor import SizingPouilletGeometryFactor
from ..components.sizing_external_stator_diameter import SizingExtStatorDiameter
from ..components.sizing_conductor_wire_section_area import SizingConductorWireSectionArea
from ..components.sizing_sm_pmsm_cg_x import SizingSMPMSMCGX
from ..components.sizing_sm_pmsm_cg_y import SizingSMPMSMCGY
from ..components.sizing_sm_pmsm_drag import SizingSMPMSMDrag
from ..components.sizing_stator_core_weight import SizingStatorCoreWeight
from ..components.sizing_rotor_weight import SizingRotorWeight
from ..components.sizing_frame_weight import SizingFrameWeight
from ..components.sizing_winding_stator_weight import SizingStatorWindingWeight
from ..components.sizing_sm_pmsm_weight import SizingMotorWeight
from ..components.sizing_sm_pmsm import SizingSMPMSM

from ..components.perf_iron_losses import PerformancesIronLosses
from ..components.perf_winding_resistivity import PerformancesWindingResistivityFixed
from ..components.perf_resistance import PerformancesResistance
from ..components.perf_joule_losses import PerformancesJouleLosses
from ..components.perf_air_dynamic_viscosity import PerformancesAirDynamicViscosity
from ..components.perf_windage_reynolds import PerformancesWindageReynolds
from ..components.perf_windage_friction_coeff import PerformancesWindageFrictionCoefficient
from ..components.perf_air_gap_windage_losses import PerformancesAirGapWindageLosses
from ..components.perf_rotor_windage_losses import PerformancesRotorWindageLoss
from ..components.perf_bearing_friction_losses import PerformancesBearingLosses
from ..components.perf_mechanical_losses import PerformancesMechanicalLosses
from ..components.perf_power_losses import PerformancesPowerLosses
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_electrical_frequency import PerformancesElectricalFrequency
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_current_rms import PerformancesCurrentRMS
from ..components.perf_phase_current_density import PerformancesPhaseCurrentDensity
from ..components.perf_surface_current_density import PerformancesSurfaceCurrentDensity
from ..components.perf_tangential_stress import PerformancesTangentialStress
from ..components.perf_air_gap_flux_density import PerformancesAirGapFluxDensity
from ..components.perf_total_flux_density import PerformancesTotalFluxDensity
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_mechanical_factor import PerformancesMechanicalFactor
from ..components.perf_sm_pmsm import PerformancesSMPMSM

from ..components.pre_lca_prod_weight_per_fu import PreLCAMotorProdWeightPerFU

from ..components.lcc_sm_pmsm_cost import LCCSMPMSMCost
from ..components.lcc_sm_pmsm_operational_cost import LCCSMPMSMOperationalCost

from ..components.cstr_enforce import (
    ConstraintsTorqueEnforce,
    ConstraintsRPMEnforce,
    ConstraintsTangentialStressEnforce,
    ConstraintsCurrentDensityEnforce,
    ConstraintsElectromagneticTorqueEnforce,
)
from ..components.cstr_ensure import (
    ConstraintsTorqueEnsure,
    ConstraintsRPMEnsure,
    ConstraintsTangentialStressEnsure,
    ConstraintsCurrentDensityEnsure,
    ConstraintsElectromagneticTorqueEnsure,
)
from ..components.cstr_sm_pmsm import ConstraintPMSMPowerRateMission
from ..constants import POSSIBLE_POSITION, DEFAULT_DYNAMIC_VISCOSITY

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_motor_new.xml"
NB_POINTS_TEST = 10


def test_bore_diameter():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorBoreDiameter(motor_id="motor_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingStatorBoreDiameter(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter", units="m"
    ) == pytest.approx(0.19, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rotor_diameter():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter",
        val=0.19,
        units="m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:air_gap_thickness",
        val=0.00075,
        units="m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingRotorDiameter(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_diameter", units="m"
    ) == pytest.approx(0.1885, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_radius_ratio():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_diameter", val=0.0988, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter",
        val=0.187,
        units="m",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingRadiusRatio(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:radius_ratio"
    ) == pytest.approx(0.528, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_active_length():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter",
        val=0.19,
        units="m",
    )
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:form_coefficient", val=0.6)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingActiveLength(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:active_length", units="m"
    ) == pytest.approx(0.3167, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_yoke_height():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorYokeHeight(motor_id="motor_1")), __file__, XML_FILE
    )

    problem = run_system(SizingStatorYokeHeight(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_yoke_height", units="m"
    ) == pytest.approx(0.0694, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_slot_width():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:conductor_slot_number", val=24)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:tooth_ratio", val=0.4407)
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter",
        val=0.19,
        units="m",
    )

    problem = run_system(SizingSlotWidth(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_width", units="m"
    ) == pytest.approx(0.0139, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_slot_height():
    ivc = get_indep_var_comp(list_inputs(SizingSlotHeight(motor_id="motor_1")), __file__, XML_FILE)

    problem = run_system(SizingSlotHeight(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_height", units="m"
    ) == pytest.approx(0.0404, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_slot_section():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_height", val=0.0392, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_width", val=0.0139, units="m"
    )

    problem = run_system(SizingSlotSectionArea(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_section_area", units="m**2"
    ) == pytest.approx(0.000545, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_conductor_section_area_per_slot_area():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_section_area",
        val=0.000545,
        units="m**2",
    )
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:slot_conductor_factor", val=1)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:slot_fill_factor", val=0.5)

    problem = run_system(SizingConductorSectionAreaPerSlot(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:conductor_section_area_per_slot",
        units="m**2",
    ) == pytest.approx(0.0002725, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_conductor_slot_number():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:number_of_phases", val=3)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:slots_per_pole_per_phase", val=2)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=2)

    problem = run_system(SizingConductorSlotNumber(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:conductor_slot_number"
    ) == pytest.approx(24, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_conductor_wire_cross_section():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1" ":conductor_section_area_per_slot",
        units="m**2",
        val=0.0002725,
    )
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:wire_per_slot", val=6)

    problem = run_system(SizingConductorWireSectionArea(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:wire_circular_section_area", units="m**2"
    ) == pytest.approx(4.54e-05, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_single_conductor_cable_length():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:active_length", val=0.3167, units="m"
    )
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:cond_twisting_coeff", val=1.25)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:end_winding_coeff", val=1.4)

    problem = run_system(SizingSingleConductorCableLength(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:conductor_cable_length"
    ) == pytest.approx(0.554225, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_pouillet_geometry_factor():
    ivc = get_indep_var_comp(
        list_inputs(SizingPouilletGeometryFactor(motor_id="motor_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingPouilletGeometryFactor(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:pouillet_geometry_factor", units="m**-1"
    ) == pytest.approx(48812.48, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_x2p_ratio():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:radius_ratio", val=0.842)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=2)

    problem = run_system(SizingRatioX2p(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:x2p_ratio"
    ) == pytest.approx(3.021, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_external_stator_diameter():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter",
        val=0.19,
        units="m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_height", val=0.0392, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_yoke_height", val=0.0396, units="m"
    )

    problem = run_system(SizingExtStatorDiameter(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_diameter"
    ) == pytest.approx(0.3476, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_stator_core_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorCoreWeight(motor_id="motor_1")), __file__, XML_FILE
    )

    problem = run_system(SizingStatorCoreWeight(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_core_mass", units="kg"
    ) == pytest.approx(28.73, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_stator_winding_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingStatorWindingWeight(motor_id="motor_1")), __file__, XML_FILE
    )

    problem = run_system(SizingStatorWindingWeight(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_winding_mass", units="kg"
    ) == pytest.approx(37.54, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rotor_weight():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=2)
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:active_length", val=0.3167, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_diameter", val=0.19, units="m"
    )

    problem = run_system(SizingRotorWeight(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_mass", units="kg"
    ) == pytest.approx(39.96, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_frame_dimension():
    ivc = get_indep_var_comp(
        list_inputs(SizingFrameDimension(motor_id="motor_1")), __file__, XML_FILE
    )

    problem = run_system(SizingFrameDimension(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:frame_length", units="m"
    ) == pytest.approx(0.48, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:frame_diameter", units="m"
    ) == pytest.approx(0.376, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_frame_weight():
    ivc = get_indep_var_comp(list_inputs(SizingFrameWeight(motor_id="motor_1")), __file__, XML_FILE)

    problem = run_system(SizingFrameWeight(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:frame_mass", units="kg"
    ) == pytest.approx(23.9, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_SMPMSM_weight():
    ivc = get_indep_var_comp(list_inputs(SizingMotorWeight(motor_id="motor_1")), __file__, XML_FILE)

    problem = run_system(SizingMotorWeight(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mass", units="kg"
    ) == pytest.approx(262.91, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_motor_cg_x():
    expected_cg = [2.59, 0.25]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingSMPMSMCGX(motor_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingSMPMSMCGX(motor_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_cg_y():
    expected_cg = [1.5, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingSMPMSMCGY(motor_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingSMPMSMCGY(motor_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_drag():
    expected_drag_ls = [0.66, 0.0]
    expected_drag_cruise = [0.653, 0.0]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_drag_ls, expected_drag_cruise
    ):
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingSMPMSMDrag(motor_id="motor_1", position=option, low_speed_aero=ls_option)
                ),
                __file__,
                XML_FILE,
            )
            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingSMPMSMDrag(motor_id="motor_1", position=option, low_speed_aero=ls_option), ivc
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
    problem = run_system(ConstraintsTorqueEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(856.63, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_electromagnetic_torque_enforce():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:electromagnetic_torque_max",
        val=856.6,
        units="N*m",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsElectromagneticTorqueEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:electromagnetic_torque_rating", units="N*m"
    ) == pytest.approx(856.63, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_rpm_enforce():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rpm_max", val=4500.0, units="min**-1"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRPMEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(4500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_tangential_enforce():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:tangential_stress_max",
        val=156.2,
        units="MPa",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTangentialStressEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:tangential_stress_caliber", units="MPa"
    ) == pytest.approx(156.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_current_density_enforce():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:current_density_ac_max",
        val=156.2,
        units="A/m**2",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsCurrentDensityEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:current_density_ac_caliber", units="A/m**2"
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
    problem = run_system(ConstraintsTorqueEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:SM_PMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_electromagnetic_torque_ensure():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:electromagnetic_torque_max",
        val=856.6,
        units="N*m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:electromagnetic_torque_rating",
        val=856.6,
        units="N*m",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsElectromagneticTorqueEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:SM_PMSM:motor_1:electromagnetic_torque_rating",
        units="N*m",
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
    problem = run_system(ConstraintsRPMEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:SM_PMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(-500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_tangential_ensure():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:tangential_stress_max",
        val=156.2,
        units="MPa",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:tangential_stress_caliber",
        val=200.0,
        units="MPa",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTangentialStressEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:SM_PMSM:motor_1:tangential_stress_caliber",
        units="MPa",
    ) == pytest.approx(-43.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_current_density_ensure():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:current_density_ac_max",
        val=156.2,
        units="A/m**2",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:current_density_ac_caliber",
        val=200.0,
        units="A/m**2",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsCurrentDensityEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:SM_PMSM:motor_1:current_density_ac_caliber",
        units="A/m**2",
    ) == pytest.approx(-43.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:shaft_power_max", val=1432.599, units="kW"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintPMSMPowerRateMission(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:shaft_power_rating", units="kW"
    ) == pytest.approx(1432.599, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistivity_fixed():
    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:winding_temperature", val=180, units="degC"
    )

    problem = run_system(
        PerformancesWindingResistivityFixed(motor_id="motor_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:resistivity",
        units="ohm*m",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 2.736384e-08), rel=1e-3)

    problem.check_partials(compact_print=True)


def test_resistance():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:pouillet_geometry_factor",
        val=48812.48,
        units="m**-1",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:resistivity",
        val=2.736384e-08,
        units="ohm*m",
        shape=NB_POINTS_TEST,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesResistance(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:resistance", units="ohm"
    ) == pytest.approx(
        np.full(NB_POINTS_TEST, 0.0013357),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_joule_losses():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:resistance",
        val=np.full(NB_POINTS_TEST, 0.0013357),
        units="ohm",
    )

    ivc.add_output("ac_current_rms_in_one_phase", 1970.84 * np.ones(NB_POINTS_TEST), units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesJouleLosses(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("joule_power_losses", units="W") == pytest.approx(
        15564.42 * np.ones(NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_frequency():
    ivc = om.IndepVarComp()

    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=2)

    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesElectricalFrequency(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("electrical_frequency", units="Hz") == pytest.approx(
        np.full(NB_POINTS_TEST, 532.33), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_tangential_stress():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTangentialStress(motor_id="motor_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "electromagnetic_torque",
        np.array([201.89, 230.53, 259.05, 287.58, 316.21, 344.74, 374.00, 402.95, 431.16, 459.68]),
        units="N*m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTangentialStress(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("tangential_stress", units="MPa") == pytest.approx(
        np.array(
            [
                0.00456,
                0.0052,
                0.00584,
                0.00649,
                0.00713,
                0.00778,
                0.00844,
                0.00909,
                0.00973,
                0.01037,
            ]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_air_gap_flux_density():
    ivc = om.IndepVarComp()

    ivc.add_output("surface_current_density", np.full(NB_POINTS_TEST, 26899.72), units="A/m")
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:x2p_ratio", 3.021)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:tooth_ratio", 0.5)
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:tooth_flux_density_ratio", 1.5)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAirGapFluxDensity(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("air_gap_flux_density", units="T") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.2576),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_phase_current_density():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:wire_circular_section_area",
        4.54e-05,
        units="m**2",
    )

    ivc.add_output("ac_current_rms_in_one_phase", 80.0 * np.ones(NB_POINTS_TEST), units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPhaseCurrentDensity(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("ac_phase_current_density", units="A/m**2") == pytest.approx(
        np.full(NB_POINTS_TEST, 1762114.54), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_surface_current_density():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesSurfaceCurrentDensity(motor_id="motor_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output("ac_current_rms_in_one_phase", 80.0 * np.ones(NB_POINTS_TEST), units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSurfaceCurrentDensity(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("surface_current_density", units="A/m") == pytest.approx(
        np.full(NB_POINTS_TEST, 16767.49), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_total_flux_density():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "air_gap_flux_density",
        np.array([0.92, 1.05, 1.18, 1.31, 1.44, 1.57, 1.70, 1.83, 1.96, 2.00]),
        units="T",
    )

    ivc.add_output("surface_current_density", np.full(NB_POINTS_TEST, 26474.99), units="A/m")
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:x2p_ratio", 3.021)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTotalFluxDensity(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("total_flux_density", units="T") == pytest.approx(
        np.array([0.93, 1.06, 1.18, 1.31, 1.44, 1.57, 1.70, 1.83, 1.96, 2.00]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_iron_losses():
    ivc = om.IndepVarComp()

    ivc.add_output("electrical_frequency", np.full(NB_POINTS_TEST, 532.33), units="s**-1")
    ivc.add_output("air_gap_flux_density", np.full(NB_POINTS_TEST, 0.9), units="T")
    ivc.add_output("data:propulsion:he_power_train:SM_PMSM:motor_1:mass", val=262.91, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesIronLosses(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("iron_power_losses", units="W") == pytest.approx(
        np.full(NB_POINTS_TEST, 4238.2), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_air_dynamic_viscosity():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "altitude",
        units="m",
        val=np.zeros(NB_POINTS_TEST),
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAirDynamicViscosity(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("dynamic_viscosity", units="kg/m/s") == pytest.approx(
        np.full(NB_POINTS_TEST, DEFAULT_DYNAMIC_VISCOSITY), rel=1e-2
    )
    problem.check_partials(compact_print=True)


def test_windage_reynolds():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_diameter", val=0.16, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:air_gap_thickness", val=0.015, units="m"
    )
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")
    ivc.add_output(
        "dynamic_viscosity", DEFAULT_DYNAMIC_VISCOSITY * np.ones(NB_POINTS_TEST), units="kg/m/s"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesWindageReynolds(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("air_gap_reynolds_number") == pytest.approx(
        np.full(NB_POINTS_TEST, 137385.27), rel=1e-2
    )
    assert problem.get_val("rotor_end_reynolds_number") == pytest.approx(
        np.full(NB_POINTS_TEST, 732721.46), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_windage_friction_coefficient():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesWindageFrictionCoefficient(
                motor_id="motor_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output("air_gap_reynolds_number", np.full(NB_POINTS_TEST, 137385.27))
    ivc.add_output("rotor_end_reynolds_number", np.full(NB_POINTS_TEST, 732721.46))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesWindageFrictionCoefficient(motor_id="motor_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("air_gap_friction_coeff") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.000623), rel=1e-2
    )
    assert problem.get_val("rotor_end_friction_coeff") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.009803), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_air_gap_windage_losses():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesAirGapWindageLosses(motor_id="motor_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output("air_gap_friction_coeff", np.full(NB_POINTS_TEST, 0.001846))
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAirGapWindageLosses(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("air_gap_windage_losses", units="W") == pytest.approx(
        7310.52 * np.ones(NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_rotor_windage_losses():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesRotorWindageLoss(motor_id="motor_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output("rotor_end_friction_coeff", np.full(NB_POINTS_TEST, 0.009803))
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesRotorWindageLoss(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("rotor_windage_losses", units="W") == pytest.approx(
        6507.18 * np.ones(NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_bearing_friction_losses():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesBearingLosses(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )

    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesBearingLosses(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("bearing_friction_losses", units="W") == pytest.approx(
        23.4 * np.ones(NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_mechanical_losses():
    ivc = om.IndepVarComp()

    ivc.add_output("air_gap_windage_losses", 603.46 * np.ones(NB_POINTS_TEST), units="W")
    ivc.add_output("rotor_windage_losses", 287.91 * np.ones(NB_POINTS_TEST), units="W")
    ivc.add_output("bearing_friction_losses", 23.4 * np.ones(NB_POINTS_TEST), units="W")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesMechanicalLosses(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("mechanical_power_losses", units="W") == pytest.approx(
        1226.08 * np.ones(NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_power_losses():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "mechanical_power_losses",
        1226.08 * np.ones(NB_POINTS_TEST),
        units="W",
    )
    ivc.add_output(
        "iron_power_losses",
        4238.2 * np.ones(NB_POINTS_TEST),
        units="W",
    )
    ivc.add_output(
        "joule_power_losses",
        15564.42 * np.ones(NB_POINTS_TEST),
        units="W",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesPowerLosses(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("power_losses", units="W") == pytest.approx(
        21028.7 * np.ones(NB_POINTS_TEST), rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_efficiency():
    ivc = om.IndepVarComp()

    ivc.add_output("shaft_power_out", 1432.6 * np.ones(NB_POINTS_TEST), units="kW")
    ivc.add_output("power_losses", 21028.7 * np.ones(NB_POINTS_TEST), units="W")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEfficiency(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("efficiency") == pytest.approx(
        0.9846 * np.ones(NB_POINTS_TEST), rel=1e-3
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
        PerformancesApparentPower(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
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
        PerformancesCurrentRMS(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("ac_current_rms_in", units="A") == pytest.approx(
        [190.17, 211.04, 231.91, 250.46, 269.02, 285.25, 301.48, 315.4, 331.63, 345.55],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "torque_out",
        np.array([82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0]),
        units="N*m",
    )
    ivc.add_output(
        "electromagnetic_torque",
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
    ivc.add_output("ac_phase_current_density", np.linspace(1.9e6, 2.0e6, 10), units="A/m**2")
    ivc.add_output(
        "tangential_stress",
        np.array([0.0119, 0.0136, 0.0153, 0.017, 0.0187, 0.0203, 0.0221, 0.0237, 0.0254, 0.0271]),
        units="MPa",
    )
    ivc.add_output("air_gap_flux_density", np.linspace(0.8, 0.9, 10), units="T")
    ivc.add_output("total_flux_density", np.linspace(0.8, 1.5, 10), units="T")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
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
        "data:propulsion:he_power_train:SM_PMSM:motor_1:electromagnetic_torque_max", units="N*m"
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
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:current_density_ac_max", units="A/m**2"
    ) == pytest.approx(2.0e6, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:air_gap_flux_density_max", units="T"
    ) == pytest.approx(0.9, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:total_flux_density_max", units="T"
    ) == pytest.approx(1.5, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:tangential_stress_max", units="MPa"
    ) == pytest.approx(0.0271, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_mechanical_factor():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesMechanicalFactor(motor_id="motor_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesMechanicalFactor(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mechanical_factor"
    ) == pytest.approx(0.636, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performance_SM_PMSM():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesSMPMSM(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )

    ivc.add_output("shaft_power_out", 1432.6 * np.ones(NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm", 15970 * np.ones(NB_POINTS_TEST), units="min**-1")
    ivc.add_output("altitude", val=np.zeros(NB_POINTS_TEST), units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSMPMSM(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
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
    ivc = get_indep_var_comp(list_inputs(SizingSMPMSM(motor_id="motor_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingSMPMSM(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:bore_diameter", units="m"
    ) == pytest.approx(0.19, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_diameter", units="m"
    ) == pytest.approx(0.1885, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:active_length", units="m"
    ) == pytest.approx(0.317, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_yoke_height", units="m"
    ) == pytest.approx(0.044, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_width"
    ) == pytest.approx(0.0139, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:slot_height", units="m"
    ) == pytest.approx(0.0404, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_diameter", units="m"
    ) == pytest.approx(0.359, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_core_mass", units="kg"
    ) == pytest.approx(153.5, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:stator_winding_mass", units="kg"
    ) == pytest.approx(38.85, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:rotor_mass", units="kg"
    ) == pytest.approx(39.5, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:frame_mass", units="kg"
    ) == pytest.approx(24.9, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:frame_diameter", units="m"
    ) == pytest.approx(0.388, rel=1e-2)
    problem.check_partials(compact_print=True)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mass", units="kg"
    ) == pytest.approx(256.8, rel=1e-2)

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
    problem = run_system(PreLCAMotorProdWeightPerFU(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:mass_per_fu", units="kg"
    ) == pytest.approx(5.258e-4, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cost():
    ivc = get_indep_var_comp(list_inputs(LCCSMPMSMCost(motor_id="motor_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCSMPMSMCost(motor_id="motor_1"), ivc)

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
    problem = run_system(LCCSMPMSMOperationalCost(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:SM_PMSM:motor_1:operational_cost", units="USD/yr"
    ) == pytest.approx(1634.74, rel=1e-2)

    problem.check_partials(compact_print=True)
