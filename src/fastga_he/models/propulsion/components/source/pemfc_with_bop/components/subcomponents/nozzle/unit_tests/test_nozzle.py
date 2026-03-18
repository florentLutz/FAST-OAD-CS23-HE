import pytest
import numpy as np
import openmdao.api as om
import os.path as pth

from ..sizing_nozzle_length import SizingNozzleLength
from ..sizing_nozzle_exit_area import SizingNozzleExitArea
from ..sizing_nozzle_exit_dimension import SizingNozzleExitDimension
from ..sizing_outer_dimension import SizingOuterDimension
from ..sizing_inner_volume import SizingInnerVolume
from ..sizing_outer_volume import SizingOuterVolume
from ..sizing_nozzle_weight import SizingNozzleWeight
from ..sizing_entry_hydraulic_diameter import SizingEntryHydraulicDiameter
from ..sizing_nozzle_relative_roughness import SizingNozzleRelativeRoughness
from ..sizing_nozzle_exit_height_length_ratio import SizingNozzleExitHeightLengthRatio
from ..sizing_nozzle_alpha_angle import SizingNozzleAlphaAngle
from ..sizing_nozzle import SizingNozzle

from ..perf_nozzle_air_speed import PerformancesNozzleAirSpeed
from ..perf_nozzle_inlet_pressure import PerformancesNozzleInletPressure
from ..perf_nozzle_reynolds_number import PerformancesNozzleReynoldsNumber
from ..perf_nozzle_darcy_friction_factor import PerformancesNozzleDarcyFrictionFactor
from ..perf_nozzle_contraction_loss_coeff import PerformancesNozzleContractionLossCoefficient
from ..perf_nozzle_friction_loss_coeff import PerformancesNozzleFrictionLossCoefficient
from ..perf_nozzle_pressure_drop import PerformancesNozzlePressureDrop
from ..perf_nozzle_drag import PerformancesNozzleDrag
from ..perf_nozzle import PerformancesNozzle

from tests.testing_utilities import run_system

NB_POINTS_TEST = 10


def test_sizing_nozzle_length():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length",
        units="m",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
        val=0.0907,
    )

    problem = run_system(
        SizingNozzleLength(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            connected_diffuser_id="diffuser_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:length",
        units="m",
    ) == pytest.approx(3.41, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_nozzle_exit_area():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.0907,
    )

    problem = run_system(
        SizingNozzleExitArea(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_area",
        units="m**2",
    ) == pytest.approx(0.0326, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_nozzle_exit_dimension():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_area",
        units="m**2",
        val=0.0326,
    )

    problem = run_system(
        SizingNozzleExitDimension(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_width",
        units="m",
    ) == pytest.approx(0.3611, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_height",
        units="m",
    ) == pytest.approx(0.0903, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_outer_dimension():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_height",
        units="m",
        val=0.0903,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_width",
        units="m",
        val=0.3611,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:wall_thickness",
        units="m",
        val=0.001,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,
    )

    problem = run_system(
        SizingOuterDimension(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            connected_heat_exchanger_id="heat_exchanger_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:outlet_side_height",
        units="m",
    ) == pytest.approx(0.0923, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:outlet_side_width",
        units="m",
    ) == pytest.approx(0.3631, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:inlet_side_height",
        units="m",
    ) == pytest.approx(0.721, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:inlet_side_width",
        units="m",
    ) == pytest.approx(0.0927, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_inner_volume():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:length",
        units="m",
        val=3.41,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_height",
        units="m",
        val=0.0903,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_width",
        units="m",
        val=0.3611,
    )

    problem = run_system(
        SizingInnerVolume(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            connected_heat_exchanger_id="heat_exchanger_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:inner_volume",
        units="m**3",
    ) == pytest.approx(0.263, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_outer_volume():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:length",
        units="m",
        val=3.41,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1"
        ":inlet_side_height",
        units="m",
        val=0.721,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1"
        ":inlet_side_width",
        units="m",
        val=0.0927,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:outlet_side_height",
        units="m",
        val=0.0923,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:outlet_side_width",
        units="m",
        val=0.3631,
    )

    problem = run_system(
        SizingOuterVolume(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:outer_volume",
        units="m**3",
    ) == pytest.approx(0.268, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_nozzle_weight():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:inner_volume",
        units="m**3",
        val=0.263,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:outer_volume",
        units="m**3",
        val=0.268,
    )

    problem = run_system(
        SizingNozzleWeight(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:mass",
        units="kg",
    ) == pytest.approx(13.5, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_entry_hydraulic_diameter():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.0907,
    )

    problem = run_system(
        SizingEntryHydraulicDiameter(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:entry_hydraulic_diameter",
        units="m",
    ) == pytest.approx(0.161, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_nozzle_relative_roughness():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:entry_hydraulic_diameter",
        units="m",
        val=0.161,
    )

    problem = run_system(
        SizingNozzleRelativeRoughness(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:relative_roughness",
    ) == pytest.approx(6.21e-4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_nozzle_exit_height_length_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_height",
        units="m",
        val=0.0903,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:length",
        units="m",
        val=3.41,
    )

    problem = run_system(
        SizingNozzleExitHeightLengthRatio(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_height_length_ratio",
    ) == pytest.approx(0.6, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_nozzle_alpha_angle():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:length",
        units="m",
        val=3.41,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_width",
        units="m",
        val=0.3611,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.0907,
    )

    problem = run_system(
        SizingNozzleAlphaAngle(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            connected_heat_exchanger_id="heat_exchanger_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:alpha",
        units="deg",
    ) == pytest.approx(-2.27, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_nozzle():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length",
        units="m",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:wall_thickness",
        units="m",
        val=0.001,
    )

    problem = run_system(
        SizingNozzle(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            connected_diffuser_id="diffuser_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:length",
        units="m",
    ) == pytest.approx(3.41, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:mass",
        units="kg",
    ) == pytest.approx(11.65, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_nozzle_air_speed():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:area_ratio",
        units="unitless",
        val=2.0,
    )
    ivc.add_output("entry_air_speed", val=80.0, units="m/s", shape=NB_POINTS_TEST)

    problem = run_system(
        PerformancesNozzleAirSpeed(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "average_air_speed",
        units="m/s",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 120.0), rel=1e-2)

    assert problem.get_val(
        "exit_air_speed",
        units="m/s",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 160.0), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_nozzle_inlet_pressure():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_pressure_drop",
        units="Pa",
        val=500.0,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "diffuser_exit_pressure",
        units="Pa",
        val=100000.0,
        shape=NB_POINTS_TEST,
    )

    problem = run_system(
        PerformancesNozzleInletPressure(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "nozzle_inlet_pressure",
        units="Pa",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 99500.0), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_nozzle_reynolds_number():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:entry_hydraulic_diameter",
        units="m",
        val=0.161,
    )
    ivc.add_output(
        "average_air_speed",
        val=120.0,
        units="m/s",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "nozzle_air_density",
        val=1.225,
        units="kg/m**3",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "nozzle_air_dynamic_viscosity",
        val=1.81e-5,
        units="Pa*s",
        shape=NB_POINTS_TEST,
    )

    problem = run_system(
        PerformancesNozzleReynoldsNumber(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "air_reynolds_number",
        units="unitless",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 1307569.06), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_nozzle_darcy_friction_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "air_reynolds_number",
        val=1307569.06,
        units="unitless",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:relative_roughness",
        val=6.21e-4,
        units="unitless",
    )

    problem = run_system(
        PerformancesNozzleDarcyFrictionFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "nozzle_darcy_friction_factor",
        units="unitless",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 0.018), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_nozzle_contraction_loss_coeff():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_height_length_ratio",
        val=0.6,
        units="unitless",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:alpha",
        val=3.0,
        units="deg",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:area_ratio",
        val=2.0,
        units="unitless",
    )

    problem = run_system(
        PerformancesNozzleContractionLossCoefficient(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "nozzle_contraction_loss_coefficient", units="unitless"
    ) == pytest.approx(0.182, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_nozzle_friction_loss_coeff():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "nozzle_darcy_friction_factor",
        val=0.018,
        units="unitless",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:alpha",
        val=3.0,
        units="deg",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:area_ratio",
        val=2.0,
        units="unitless",
    )

    problem = run_system(
        PerformancesNozzleFrictionLossCoefficient(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val("nozzle_friction_loss_coefficient", units="unitless") == pytest.approx(
        0.0322, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_perf_nozzle_pressure_drop():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "nozzle_friction_loss_coefficient",
        val=0.0322,
        units="unitless",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "nozzle_contraction_loss_coefficient",
        val=0.182,
        units="unitless",
    )
    ivc.add_output(
        "average_air_speed",
        val=120.0,
        units="m/s",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "nozzle_air_density",
        val=1.225,
        units="kg/m**3",
        shape=NB_POINTS_TEST,
    )

    problem = run_system(
        PerformancesNozzlePressureDrop(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:air_pressure_drop",
        units="Pa",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 1889.24), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_nozzle_drag():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("exit_air_speed", val=160.0, units="m/s", shape=NB_POINTS_TEST)
    ivc.add_output("true_air_speed", val=80.0, units="m/s", shape=NB_POINTS_TEST)
    ivc.add_output("air_mass_flow_rate", val=0.5, units="kg/s", shape=NB_POINTS_TEST)

    problem = run_system(
        PerformancesNozzleDrag(
            number_of_points=NB_POINTS_TEST,
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:drag", units="N"
    ) == pytest.approx(np.full(NB_POINTS_TEST, 40.0), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_nozzle():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:area_ratio",
        units="unitless",
        val=2.0,
    )
    ivc.add_output(
        "entry_air_speed",
        val=80.0,
        units="m/s",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_pressure_drop",
        units="Pa",
        val=500.0,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "diffuser_exit_pressure",
        units="Pa",
        val=100000.0,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:entry_hydraulic_diameter",
        units="m",
        val=0.161,
    )
    ivc.add_output(
        "nozzle_inlet_temperature",
        val=300.0,
        units="K",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "true_air_speed",
        val=80.0,
        units="m/s",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "air_mass_flow_rate",
        val=0.5,
        units="kg/s",
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:exit_height_length_ratio",
        val=0.6,
        units="unitless",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:alpha",
        val=3.0,
        units="deg",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:relative_roughness",
        val=6.21e-4,
        units="unitless",
    )

    problem = run_system(
        PerformancesNozzle(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            nozzle_id="nozzle_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    # Research expected output value in .xml file
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:nozzle_1:drag", units="N"
    ) == pytest.approx(np.full(NB_POINTS_TEST, 40.0), rel=1e-2)

    problem.check_partials(compact_print=True)
