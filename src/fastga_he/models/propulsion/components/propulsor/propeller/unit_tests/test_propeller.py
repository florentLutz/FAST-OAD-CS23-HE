# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.sizing_weight import SizingPropellerWeight
from ..components.sizing_propeller_depth import SizingPropellerDepth
from ..components.sizing_propeller_cg import SizingPropellerCG
from ..components.sizing_propeller_ref_cl import SizingPropellerReferenceCl
from ..components.sizing_propeller_ref_chord import SizingPropellerReferenceChord
from ..components.sizing_propeller_radius_to_span_ratio import SizingPropellerDiameterToSpanRatio
from ..components.sizing_propeller_radius_to_chord_ratio import SizingPropellerDiameterToChordRatio
from ..components.sizing_propeller_flapped_span_ratio import SizingPropellerFlappedRatio
from ..components.sizing_propeller_wing_ac_distance import SizingPropellerWingACDistance
from ..components.sizing_propeller_wing_le_distance_ratio import SizingPropellerWingLEDistanceRatio
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
from ..components.slipstream_thrust_loading import SlipstreamPropellerThrustLoading
from ..components.slipstream_axial_induction_factor import SlipstreamPropellerAxialInductionFactor
from ..components.slipstream_contraction_ratio_squared import (
    SlipstreamPropellerContractionRatioSquared,
)
from ..components.slipstream_contraction_ratio import SlipstreamPropellerContractionRatio
from ..components.slipstream_axial_induction_factor_ac import (
    SlipstreamPropellerAxialInductionFactorWingAC,
)
from ..components.slipstream_axial_induction_factor_downstream import (
    SlipstreamPropellerVelocityRatioDownstream,
)
from ..components.slipstream_height_impact_coefficients import (
    SlipstreamPropellerHeightImpactCoefficients,
)
from ..components.slipstream_height_impact import SlipstreamPropellerHeightImpact
from ..components.slipstream_lift_increase_ratio import SlipstreamPropellerLiftIncreaseRatio
from ..components.slipstream_section_lift import SlipstreamPropellerSectionLift
from ..components.slipstream_delta_cl_2d import SlipstreamPropellerDeltaCl2D
from ..components.slipstream_blown_area_ratio import SlipstreamPropellerBlownAreaRatio
from ..components.slipstream_delta_cl import SlipstreamPropellerDeltaCl
from ..components.slipstream_delta_cl_group import SlipstreamPropellerDeltaClGroup
from ..components.slipstream_delta_cd0 import SlipstreamPropellerDeltaCD0
from ..components.slipstream_delta_cm0 import SlipstreamPropellerDeltaCM0
from ..components.slipstream_delta_cm_alpha import SlipstreamPropellerDeltaCMAlpha
from ..components.slipstream_delta_cm import SlipstreamPropellerDeltaCM
from ..components.slipstream_propeller import SlipstreamPropeller
from ..components.cstr_enforce import ConstraintsTorqueEnforce
from ..components.cstr_ensure import ConstraintsTorqueEnsure

from ..components.perf_propeller import PerformancesPropeller
from ..components.sizing_propeller import SizingPropeller

from ..constants import POSSIBLE_POSITION

from stdatm import Atmosphere

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


def test_depth():

    ivc = get_indep_var_comp(
        list_inputs(SizingPropellerDepth(propeller_id="propeller_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPropellerDepth(propeller_id="propeller_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:depth", units="m"
    ) == pytest.approx(0.30, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_propeller_cg():

    expected_cg = [2.2, 0.15]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):

        ivc = get_indep_var_comp(
            list_inputs(SizingPropellerCG(propeller_id="propeller_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingPropellerCG(propeller_id="propeller_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:propeller:propeller_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_propeller_ref_cl():

    expected_cl = [1.085, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cl):

        ivc = get_indep_var_comp(
            list_inputs(SizingPropellerReferenceCl(propeller_id="propeller_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingPropellerReferenceCl(propeller_id="propeller_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:propeller:propeller_1:cl_clean_ref"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True, step=1e-7)


def test_propeller_ref_chord():

    expected_chords = [0.9275, 1.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_chords):

        ivc = get_indep_var_comp(
            list_inputs(SizingPropellerReferenceChord(propeller_id="propeller_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingPropellerReferenceChord(propeller_id="propeller_1", position=option), ivc
        )

        assert (
            problem.get_val(
                "data:propulsion:he_power_train:propeller:propeller_1:wing_chord_ref",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True, step=1e-7)


def test_diameter_to_span_ratio():

    ivc = get_indep_var_comp(
        list_inputs(SizingPropellerDiameterToSpanRatio(propeller_id="propeller_1")),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPropellerDiameterToSpanRatio(propeller_id="propeller_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:diameter_to_span_ratio"
    ) == pytest.approx(0.312, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diameter_to_chord_ratio():

    ivc = get_indep_var_comp(
        list_inputs(SizingPropellerDiameterToChordRatio(propeller_id="propeller_1")),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPropellerDiameterToChordRatio(propeller_id="propeller_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:diameter_to_chord_ratio"
    ) == pytest.approx(2.136, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_propeller_flapped_ratio():

    expected_values = [0.9072, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):

        ivc = get_indep_var_comp(
            list_inputs(SizingPropellerFlappedRatio(propeller_id="propeller_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingPropellerFlappedRatio(propeller_id="propeller_1", position=option), ivc
        )

        assert (
            problem.get_val(
                "data:propulsion:he_power_train:propeller:propeller_1:flapped_ratio",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True, step=1e-7)


def test_propeller_distance_from_wing_ac():

    expected_values = [0.341, 2.69]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):

        ivc = get_indep_var_comp(
            list_inputs(SizingPropellerWingACDistance(propeller_id="propeller_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingPropellerWingACDistance(propeller_id="propeller_1", position=option), ivc
        )

        assert (
            problem.get_val(
                "data:propulsion:he_power_train:propeller:propeller_1:from_wing_AC",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True)


def test_propeller_distance_from_wing_le_ratio():

    expected_values = [0.1617, 2.9]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):

        ivc = get_indep_var_comp(
            list_inputs(
                SizingPropellerWingLEDistanceRatio(propeller_id="propeller_1", position=option)
            ),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingPropellerWingLEDistanceRatio(propeller_id="propeller_1", position=option), ivc
        )

        assert (
            problem.get_val(
                "data:propulsion:he_power_train:propeller:propeller_1:from_wing_LE_ratio",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

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
    density = Atmosphere(altitude=np.full(NB_POINTS_TEST, 0.0)).density
    ivc.add_output("density", val=density, units="kg/m**3")
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
        np.array([0.0693, 0.0696, 0.0697, 0.0706, 0.0707, 0.0711, 0.0713, 0.0714, 0.0717, 0.0725]),
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
    density = Atmosphere(altitude=np.full(NB_POINTS_TEST, 0.0)).density
    ivc.add_output("density", val=density, units="kg/m**3")
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500), units="min**-1")
    ivc.add_output(
        "power_coefficient",
        val=np.array(
            [0.001, 0.0661, 0.0662, 0.0671, 0.0672, 0.0675, 0.0677, 0.0678, 0.0681, 0.0689]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesShaftPower(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.array([5.0, 178.8, 179.1, 181.5, 181.8, 182.6, 183.1, 183.4, 184.2, 186.4]),
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


def test_thrust_loading():

    ivc = get_indep_var_comp(
        list_inputs(
            SlipstreamPropellerThrustLoading(
                propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )

    density = Atmosphere(altitude=np.full(NB_POINTS_TEST, 0.0)).density
    ivc.add_output("density", val=density, units="kg/m**3")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerThrustLoading(
            propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("thrust_loading") == pytest.approx(
        np.array([0.0482, 0.0467, 0.0453, 0.0440, 0.0427, 0.0414, 0.0402, 0.0390, 0.0379, 0.0368]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_axial_induction_factor():

    ivc = om.IndepVarComp()

    ivc.add_output(
        "thrust_loading",
        val=np.array(
            [0.0482, 0.0467, 0.0453, 0.0440, 0.0427, 0.0414, 0.0402, 0.0390, 0.0379, 0.0368]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerAxialInductionFactor(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("axial_induction_factor") == pytest.approx(
        np.array([0.0298, 0.0289, 0.0281, 0.0273, 0.0265, 0.0257, 0.025, 0.0242, 0.0236, 0.0229]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_contraction_ratio_square():

    ivc = get_indep_var_comp(
        list_inputs(
            SlipstreamPropellerContractionRatioSquared(
                propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "axial_induction_factor",
        val=np.array(
            [0.0298, 0.0289, 0.0281, 0.0273, 0.0265, 0.0257, 0.025, 0.0242, 0.0236, 0.0229]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerContractionRatioSquared(
            propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("contraction_ratio_squared") == pytest.approx(
        np.array([0.9907, 0.9909, 0.9912, 0.9914, 0.9917, 0.9919, 0.9921, 0.9924, 0.9926, 0.9928]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_contraction_ratio():

    ivc = om.IndepVarComp()

    ivc.add_output(
        "contraction_ratio_squared",
        val=np.array(
            [0.9907, 0.9909, 0.9912, 0.9914, 0.9917, 0.9919, 0.9921, 0.9924, 0.9926, 0.9928]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerContractionRatio(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("contraction_ratio") == pytest.approx(
        np.array([0.9953, 0.9954, 0.9956, 0.9957, 0.9958, 0.9959, 0.996, 0.9962, 0.9963, 0.9964]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_axial_induction_factor_ac():

    ivc = om.IndepVarComp()

    ivc.add_output(
        "axial_induction_factor",
        val=np.array(
            [0.0298, 0.0289, 0.0281, 0.0273, 0.0265, 0.0257, 0.025, 0.0242, 0.0236, 0.0229]
        ),
    )
    ivc.add_output(
        "contraction_ratio_squared",
        val=np.array(
            [0.9907, 0.9909, 0.9912, 0.9914, 0.9917, 0.9919, 0.9921, 0.9924, 0.9926, 0.9928]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerAxialInductionFactorWingAC(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("axial_induction_factor_wing_ac") == pytest.approx(
        np.array([0.0394, 0.0383, 0.0372, 0.0362, 0.0350, 0.0340, 0.0331, 0.0320, 0.0312, 0.0303]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_velocity_ratio_downstream():

    ivc = om.IndepVarComp()

    ivc.add_output(
        "axial_induction_factor",
        val=np.array(
            [0.0298, 0.0289, 0.0281, 0.0273, 0.0265, 0.0257, 0.025, 0.0242, 0.0236, 0.0229]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerVelocityRatioDownstream(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("velocity_ratio_downstream") == pytest.approx(
        np.array([1.0596, 1.0578, 1.0562, 1.0546, 1.053, 1.0514, 1.05, 1.0484, 1.0472, 1.0458]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_height_impact_coefficients():

    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:propeller:propeller_1:from_wing_LE_ratio", val=0.5
    )
    ivc.add_output(
        "velocity_ratio_downstream",
        val=np.linspace(1.2, 2.3, NB_POINTS_TEST),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerHeightImpactCoefficients(
            propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("f_0") == pytest.approx(
        np.array([0.4648, 0.4494, 0.4228, 0.3958, 0.3684, 0.3404, 0.3121, 0.2833, 0.254, 0.2365]),
        rel=1e-2,
    )
    assert problem.get_val("f_1") == pytest.approx(
        np.array([1.461, 1.431, 1.387, 1.348, 1.314, 1.287, 1.266, 1.251, 1.241, 1.239]),
        rel=1e-2,
    )
    assert problem.get_val("f_2") == pytest.approx(
        np.array(
            [-1.23, -1.1911, -1.1254, -1.0599, -0.9947, -0.9297, -0.865, -0.8005, -0.7363, -0.6985]
        ),
        rel=1e-2,
    )
    assert problem.get_val("f_3") == pytest.approx(
        np.array([0.4196, 0.4043, 0.377, 0.348, 0.3173, 0.2849, 0.2507, 0.2149, 0.1773, 0.1543]),
        rel=1e-2,
    )
    assert problem.get_val("f_4") == pytest.approx(
        np.array(
            [
                -0.0515,
                -0.0494,
                -0.0457,
                -0.0415,
                -0.037,
                -0.0321,
                -0.0268,
                -0.0212,
                -0.0152,
                -0.0115,
            ]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_height_impact():

    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:propeller:propeller_1:diameter_to_chord_ratio",
        val=1.1,
    )
    ivc.add_output(
        "f_0",
        val=np.array(
            [0.4648, 0.4494, 0.4228, 0.3958, 0.3684, 0.3404, 0.3121, 0.2833, 0.254, 0.2365]
        ),
    )
    ivc.add_output(
        "f_1",
        val=np.array([1.461, 1.431, 1.387, 1.348, 1.314, 1.287, 1.266, 1.251, 1.241, 1.239]),
    )
    ivc.add_output(
        "f_2",
        val=np.array(
            [-1.23, -1.1911, -1.1254, -1.0599, -0.9947, -0.9297, -0.865, -0.8005, -0.7363, -0.6985]
        ),
    )
    ivc.add_output(
        "f_3",
        val=np.array(
            [0.4196, 0.4043, 0.377, 0.348, 0.3173, 0.2849, 0.2507, 0.2149, 0.1773, 0.1543]
        ),
    )
    ivc.add_output(
        "f_4",
        val=np.array(
            [
                -0.0515,
                -0.0494,
                -0.0457,
                -0.0415,
                -0.037,
                -0.0321,
                -0.0268,
                -0.0212,
                -0.0152,
                -0.0115,
            ]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerHeightImpact(
            propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert problem.get_val("beta") == pytest.approx(
        np.array([0.96, 0.94, 0.9, 0.87, 0.84, 0.81, 0.79, 0.76, 0.74, 0.73]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_verification_height_impact_overflow():
    """
    Verification of the implementation of the height impact effect with data from OVERFLOW
    reported in :cite:`patterson:2016`
    """

    problem = om.Problem()
    model = problem.model

    model.add_subsystem(
        name="coefficients",
        subsys=SlipstreamPropellerHeightImpactCoefficients(
            propeller_id="propeller_1", number_of_points=1
        ),
        promotes=["*"],
    )
    model.add_subsystem(
        name="height_impact",
        subsys=SlipstreamPropellerHeightImpact(propeller_id="propeller_1", number_of_points=1),
        promotes=["*"],
    )
    model.nonlinear_solver = om.NonlinearBlockGS()
    model.linear_solver = om.DirectSolver()

    problem.setup()
    problem.set_val("velocity_ratio_downstream", val=2.0)

    radius_ratios = np.array([0.35, 0.4, 0.6, 0.8, 0.7, 0.5])
    distance_ratios = np.array([0.35, 0.8, 0.6, 0.8, 0.75, 0.75])

    expected_betas = np.array([0.6114, 0.7161, 0.8095, 0.9149, 0.8646, 0.7721])

    for radius_ratio, distance_ratio, beta in zip(radius_ratios, distance_ratios, expected_betas):
        problem.set_val(
            "data:propulsion:he_power_train:propeller:propeller_1:diameter_to_chord_ratio",
            val=2.0 * radius_ratio,
        )
        problem.set_val(
            "data:propulsion:he_power_train:propeller:propeller_1:from_wing_LE_ratio",
            val=distance_ratio,
        )

        problem.run_model()

        assert problem.get_val("beta") == pytest.approx(beta, rel=2e-2)


def test_lift_increase_ratio():

    ivc = get_indep_var_comp(
        list_inputs(
            SlipstreamPropellerLiftIncreaseRatio(
                propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "axial_induction_factor_wing_ac",
        val=np.array(
            [0.0394, 0.0383, 0.0372, 0.0362, 0.0350, 0.0340, 0.0331, 0.0320, 0.0312, 0.0303]
        ),
    )
    ivc.add_output(
        "beta", val=np.array([0.96, 0.94, 0.9, 0.87, 0.84, 0.81, 0.79, 0.76, 0.74, 0.73])
    )
    ivc.add_output("alpha", val=np.full(NB_POINTS_TEST, 5.0), units="deg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerLiftIncreaseRatio(
            propeller_id="propeller_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    expected_value = np.array([2.18, 2.08, 1.94, 1.83, 1.71, 1.6, 1.52, 1.42, 1.35, 1.29])
    assert problem.get_val("lift_increase_ratio") * 100.0 == pytest.approx(expected_value, rel=2e-2)
    expected_value = np.array(
        [-11.92, -11.32, -10.49, -9.85, -9.17, -8.57, -8.12, -7.53, -7.14, -6.83]
    )
    assert problem.get_val("lift_increase_ratio_AOA_0") * 100.0 == pytest.approx(
        expected_value, rel=2e-2
    )

    problem.check_partials(compact_print=True)


def test_section_lift():

    flaps_positions = ["cruise", "landing", "takeoff"]
    expected_values = (
        np.full(NB_POINTS_TEST, 0.7157),
        np.full(NB_POINTS_TEST, 1.524),
        np.full(NB_POINTS_TEST, 1.115),
    )
    expected_values_0 = (
        np.full(NB_POINTS_TEST, 0.274),
        np.full(NB_POINTS_TEST, 1.083),
        np.full(NB_POINTS_TEST, 0.673),
    )

    for flaps_position, expected_value, expected_value_0 in zip(
        flaps_positions, expected_values, expected_values_0
    ):

        ivc = get_indep_var_comp(
            list_inputs(
                SlipstreamPropellerSectionLift(
                    number_of_points=NB_POINTS_TEST,
                    propeller_id="propeller_1",
                    flaps_position=flaps_position,
                )
            ),
            __file__,
            XML_FILE,
        )
        ivc.add_output("cl_wing_clean", val=np.full(NB_POINTS_TEST, 0.6533), units="deg")

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SlipstreamPropellerSectionLift(
                number_of_points=NB_POINTS_TEST,
                propeller_id="propeller_1",
                flaps_position=flaps_position,
            ),
            ivc,
        )

        assert problem.get_val("unblown_section_lift") == pytest.approx(expected_value, rel=1e-3)
        assert problem.get_val("unblown_section_lift_AOA_0") == pytest.approx(
            expected_value_0, rel=1e-3
        )

        problem.check_partials(compact_print=True)


def test_blown_section_lift():

    ivc = om.IndepVarComp()
    ivc.add_output("unblown_section_lift", val=np.full(NB_POINTS_TEST, 0.7157))
    ivc.add_output("unblown_section_lift_AOA_0", val=np.full(NB_POINTS_TEST, 0.274))
    ivc.add_output(
        "lift_increase_ratio",
        np.array([2.18, 2.08, 1.94, 1.83, 1.71, 1.6, 1.52, 1.42, 1.35, 1.29]) / 100.0,
    )

    problem = run_system(
        SlipstreamPropellerDeltaCl2D(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("delta_Cl_2D") == pytest.approx(
        np.array([0.0156, 0.0149, 0.0139, 0.0131, 0.0122, 0.0115, 0.0109, 0.0102, 0.0097, 0.0092]),
        rel=1e-2,
    )
    assert problem.get_val("delta_Cl_2D_AOA_0") == pytest.approx(
        np.array([0.006, 0.0057, 0.0053, 0.005, 0.0047, 0.0044, 0.0042, 0.0039, 0.0037, 0.0035]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_blown_area_ratio():

    ivc = get_indep_var_comp(
        list_inputs(
            SlipstreamPropellerBlownAreaRatio(
                number_of_points=NB_POINTS_TEST,
                propeller_id="propeller_1",
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "contraction_ratio",
        val=np.array(
            [0.9953, 0.9954, 0.9956, 0.9957, 0.9958, 0.9959, 0.996, 0.9962, 0.9963, 0.9964]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerBlownAreaRatio(
            number_of_points=NB_POINTS_TEST,
            propeller_id="propeller_1",
        ),
        ivc,
    )

    expected_value = np.array(
        [0.1441, 0.1441, 0.1441, 0.1441, 0.1442, 0.1442, 0.1442, 0.1442, 0.1442, 0.1442]
    )
    assert problem.get_val("blown_area_ratio") == pytest.approx(expected_value, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_blown_wing_lift_increase():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "blown_area_ratio",
        val=np.array(
            [0.1441, 0.1441, 0.1441, 0.1441, 0.1442, 0.1442, 0.1442, 0.1442, 0.1442, 0.1442]
        ),
    )
    ivc.add_output(
        "delta_Cl_2D",
        val=np.array(
            [0.0156, 0.0149, 0.0139, 0.0131, 0.0122, 0.0115, 0.0109, 0.0102, 0.0097, 0.0092]
        ),
    )
    ivc.add_output(
        "delta_Cl_2D_AOA_0",
        val=np.array(
            [0.006, 0.0057, 0.0053, 0.005, 0.0047, 0.0044, 0.0042, 0.0039, 0.0037, 0.0035]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerDeltaCl(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    expected_value = np.array([2.25, 2.15, 2.0, 1.89, 1.76, 1.66, 1.57, 1.47, 1.4, 1.33])
    assert problem.get_val("delta_Cl") * 1000.0 == pytest.approx(expected_value, rel=1e-2)
    expected_value_0 = np.array([0.86, 0.82, 0.76, 0.72, 0.68, 0.63, 0.61, 0.56, 0.53, 0.5])
    assert problem.get_val("delta_Cl_AOA_0") * 1000.0 == pytest.approx(expected_value_0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_blown_wing_lift_increase_group():

    ivc = get_indep_var_comp(
        list_inputs(
            SlipstreamPropellerDeltaClGroup(
                number_of_points=NB_POINTS_TEST,
                propeller_id="propeller_1",
                flaps_position="cruise",
            )
        ),
        __file__,
        XML_FILE,
    )
    density = Atmosphere(altitude=np.full(NB_POINTS_TEST, 0.0)).density
    ivc.add_output("density", val=density, units="kg/m**3")
    ivc.add_output("alpha", val=np.full(NB_POINTS_TEST, 5.0), units="deg")
    ivc.add_output("cl_wing_clean", val=np.full(NB_POINTS_TEST, 0.6533), units="deg")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerDeltaClGroup(
            number_of_points=NB_POINTS_TEST,
            propeller_id="propeller_1",
            flaps_position="cruise",
        ),
        ivc,
    )

    # om.n2(problem)

    expected_value = np.array([1.5, 1.46, 1.42, 1.38, 1.34, 1.3, 1.26, 1.23, 1.19, 1.16])
    assert problem.get_val("delta_Cl") * 1000.0 == pytest.approx(expected_value, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_blown_wing_profile_drag_increase():
    flaps_positions = ["cruise", "landing", "takeoff"]
    expected_values = (
        np.array([1.436, 1.357, 1.28, 1.212, 1.133, 1.069, 1.013, 0.947, 0.9, 0.849]),
        np.array([4.379, 4.138, 3.903, 3.696, 3.455, 3.261, 3.09, 2.888, 2.746, 2.59]),
        np.array([2.051, 1.938, 1.828, 1.731, 1.618, 1.527, 1.447, 1.353, 1.286, 1.213]),
    )

    for flaps_position, expected_value in zip(flaps_positions, expected_values):
        ivc = get_indep_var_comp(
            list_inputs(
                SlipstreamPropellerDeltaCD0(
                    number_of_points=NB_POINTS_TEST,
                    propeller_id="propeller_1",
                    flaps_position=flaps_position,
                )
            ),
            __file__,
            XML_FILE,
        )
        ivc.add_output(
            "axial_induction_factor_wing_ac",
            val=np.array(
                [0.0394, 0.0383, 0.0372, 0.0362, 0.0350, 0.0340, 0.0331, 0.0320, 0.0312, 0.0303]
            ),
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SlipstreamPropellerDeltaCD0(
                number_of_points=NB_POINTS_TEST,
                propeller_id="propeller_1",
                flaps_position=flaps_position,
            ),
            ivc,
        )

        assert problem.get_val("delta_Cd") * 1e6 == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_blown_wing_pitching_moment_increase():

    flaps_positions = ["cruise", "landing", "takeoff"]
    expected_values = (
        np.array([-7.591, -7.173, -6.767, -6.408, -5.99, -5.653, -5.357, -5.007, -4.76, -4.489]),
        np.array([-85.97, -81.23, -76.63, -72.57, -67.84, -64.02, -60.67, -56.71, -53.91, -50.84]),
        np.array([-46.26, -43.72, -41.24, -39.05, -36.51, -34.45, -32.65, -30.52, -29.01, -27.36]),
    )

    for flaps_position, expected_value in zip(flaps_positions, expected_values):
        ivc = get_indep_var_comp(
            list_inputs(
                SlipstreamPropellerDeltaCM0(
                    number_of_points=NB_POINTS_TEST,
                    propeller_id="propeller_1",
                    flaps_position=flaps_position,
                )
            ),
            __file__,
            XML_FILE,
        )
        ivc.add_output(
            "axial_induction_factor_wing_ac",
            val=np.array(
                [0.0394, 0.0383, 0.0372, 0.0362, 0.0350, 0.0340, 0.0331, 0.0320, 0.0312, 0.0303]
            ),
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SlipstreamPropellerDeltaCM0(
                number_of_points=NB_POINTS_TEST,
                propeller_id="propeller_1",
                flaps_position=flaps_position,
            ),
            ivc,
        )

        assert problem.get_val("delta_Cm0") * 1e6 == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_blown_wing_delta_cm_alpha():

    flaps_positions = ["cruise", "landing", "takeoff"]
    expected_values = (
        np.zeros(NB_POINTS_TEST),
        np.linspace(-0.00221667, -0.00158333, NB_POINTS_TEST),
        np.linspace(-0.00105, -0.00075, NB_POINTS_TEST),
    )

    for flaps_position, expected_value in zip(flaps_positions, expected_values):
        ivc = get_indep_var_comp(
            list_inputs(
                SlipstreamPropellerDeltaCMAlpha(
                    number_of_points=NB_POINTS_TEST,
                    flaps_position=flaps_position,
                )
            ),
            __file__,
            XML_FILE,
        )
        ivc.add_output(
            "delta_Cl",
            val=np.linspace(0.15, 0.12, NB_POINTS_TEST),
        )
        ivc.add_output(
            "delta_Cl_AOA_0",
            val=np.linspace(0.08, 0.07, NB_POINTS_TEST),
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SlipstreamPropellerDeltaCMAlpha(
                number_of_points=NB_POINTS_TEST,
                flaps_position=flaps_position,
            ),
            ivc,
        )

        assert problem.get_val("delta_Cm_alpha") == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_blown_wing_delta_cm():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "delta_Cm0",
        val=np.array(
            [-46.26, -43.72, -41.24, -39.05, -36.51, -34.45, -32.65, -30.52, -29.01, -27.36]
        )
        * 1e-6,
    )
    ivc.add_output(
        "delta_Cm_alpha",
        val=np.linspace(-0.00105, -0.00075, NB_POINTS_TEST),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamPropellerDeltaCM(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    expected_value = np.array(
        [
            -0.00109626,
            -0.00106039,
            -0.00102457,
            -0.00098905,
            -0.00095318,
            -0.00091778,
            -0.00088265,
            -0.00084719,
            -0.00081234,
            -0.00077736,
        ]
    )
    assert problem.get_val("delta_Cm") == pytest.approx(expected_value, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_slipstream_propeller():

    flaps_positions = ["cruise", "landing"]
    expected_delta_cds = (
        np.array([1.2371, 1.1654, 1.0983, 1.0355, 0.9766, 0.9213, 0.8695, 0.8209, 0.7752, 0.7323])
        * 1e-6,
        np.array([3.7724, 3.5539, 3.3493, 3.1576, 2.978, 2.8096, 2.6516, 2.5033, 2.364, 2.2332])
        * 1e-6,
    )
    expected_delta_cls = (
        np.array([1.5028, 1.4595, 1.4176, 1.3772, 1.3382, 1.3005, 1.264, 1.2287, 1.1946, 1.1616])
        * 1e-3,
        np.array([3.2016, 3.1094, 3.0202, 2.9341, 2.851, 2.7706, 2.6929, 2.6177, 2.545, 2.4747])
        * 1e-3,
    )
    expected_delta_cms = (
        np.array(
            [
                -6.5397,
                -6.1609,
                -5.8062,
                -5.4739,
                -5.1625,
                -4.8705,
                -4.5966,
                -4.3396,
                -4.0982,
                -3.8714,
            ]
        )
        * 1e-6,
        np.array(
            [
                -74.0641,
                -69.7733,
                -65.7562,
                -61.9933,
                -58.4668,
                -55.1602,
                -52.0581,
                -49.1466,
                -46.4126,
                -43.8442,
            ]
        )
        * 1e-6,
    )

    for flaps_position, expected_delta_cd, expected_delta_cl, expected_delta_cm in zip(
        flaps_positions, expected_delta_cds, expected_delta_cls, expected_delta_cms
    ):

        ivc = get_indep_var_comp(
            list_inputs(
                SlipstreamPropeller(
                    number_of_points=NB_POINTS_TEST,
                    propeller_id="propeller_1",
                    flaps_position=flaps_position,
                )
            ),
            __file__,
            XML_FILE,
        )

        density = Atmosphere(altitude=np.full(NB_POINTS_TEST, 0.0)).density
        ivc.add_output("density", val=density, units="kg/m**3")
        ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")
        ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
        ivc.add_output("alpha", val=np.full(NB_POINTS_TEST, 5.0), units="deg")
        ivc.add_output("cl_wing_clean", val=np.full(NB_POINTS_TEST, 0.6533), units="deg")

        problem = run_system(
            SlipstreamPropeller(
                number_of_points=NB_POINTS_TEST,
                propeller_id="propeller_1",
                flaps_position=flaps_position,
            ),
            ivc,
        )

        assert problem.get_val("delta_Cd") == pytest.approx(expected_delta_cd, rel=1e-2)
        assert problem.get_val("delta_Cl") == pytest.approx(expected_delta_cl, rel=1e-2)
        assert problem.get_val("delta_Cm") == pytest.approx(expected_delta_cm, rel=1e-2)


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
    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:depth", units="m"
    ) == pytest.approx(0.30, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:CG:x", units="m"
    ) == pytest.approx(2.2, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:low_speed:CD0"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True, step=1e-7)


def test_propeller_performances():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesPropeller(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    density = Atmosphere(altitude=np.full(NB_POINTS_TEST, 0.0)).density
    ivc.add_output("density", val=density, units="kg/m**3")
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPropeller(propeller_id="propeller_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.array([187.4, 188.2, 188.5, 191.1, 191.4, 192.2, 192.7, 193.1, 193.9, 196.2]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)
