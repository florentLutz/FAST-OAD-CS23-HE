import pytest
import numpy as np
import openmdao.api as om
import os.path as pth


from ..perf_diffuser_exit_air_speed import PerformancesDiffuserExitAirSpeed
from ..perf_diffuser_average_air_speed import PerformancesDiffuserAverageAirSpeed
from ..perf_diffuser_first_stall_angles import PerformancesDiffuserFirstStallAngel
from ..perf_diffuser_appreciable_stall_angles import PerformancesDiffuserAppreciableStallAngles
from ..perf_diffuser_reynolds_number import PerformancesDiffuserReynoldsNumber
from ..perf_diffuser_darcy_friction_factor import PerformancesDiffuserDarcyFrictionFactor
from ..perf_diffuser_singular_pressure_loss_coeff import (
    PerformancesDiffuserSingularPressureLossCoefficient,
)
from ..perf_diffuser_expansion_loss_coeff import PerformancesDiffuserExpansionLossCoefficient
from ..perf_diffuser_friction_loss_coeff import PerformancesDiffuserFrictionLossCoefficient
from ..perf_diffuser_pressure_drop import PerformancesDiffuserPressureDrop
from ..perf_diffuser_exit_total_pressure import PerformancesDiffuserExitTotalPressure
from ..perf_diffuser_exit_total_temperature import PerformancesDiffuserExitTotalTemperature
from ..perf_diffuser import PerformancesDiffuser

from ..sizing_diffuser_angles import SizingDiffuserAngles
from ..sizing_outer_dimension import SizingOuterDimension
from ..sizing_inner_volume import SizingInnerVolume
from ..sizing_outer_volume import SizingOuterVolume
from ..sizing_diffuser_weight import SizingDiffuserWeight
from ..sizing_cross_section_area import SizingCrossSectionArea
from ..sizing_area_ratio import SizingAreaRatio
from ..sizing_diffuser_stall_check_ratios import SizingDiffuserStallCheckRatios
from ..sizing_entry_hydraulic_diameter import SizingEntryHydraulicDiameter
from ..sizing_diffuser_relative_roughness import SizingDiffuserRelativeRoughness
from ..sizing_diffuser import SizingDiffuser

from tests.testing_utilities import run_system

NB_POINTS_TEST = 10


def test_diffuser_angles():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length",
        units="m",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":coolant_flow_length",
        units="m",
        val=0.05,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":highlight_width",
        units="m",
        val=0.1592,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingDiffuserAngles(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            connected_air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1" ":diffuser_1:alpha",
        units="deg",
    ) == pytest.approx(18.76, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1" ":diffuser_1:beta",
        units="deg",
    ) == pytest.approx(-3.13, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_outer_dimension():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:wall_thickness",
        units="m",
        val=0.01,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":coolant_flow_length",
        units="m",
        val=0.05,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":highlight_width",
        units="m",
        val=0.1592,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingOuterDimension(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            connected_air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1"
        ":diffuser_1:inlet_side_height",
        units="m",
    ) == pytest.approx(0.0598, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1"
        ":diffuser_1:outlet_side_height",
        units="m",
    ) == pytest.approx(0.739, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1"
        ":diffuser_1:outlet_side_width",
        units="m",
    ) == pytest.approx(0.07, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1"
        ":diffuser_1:inlet_side_width",
        units="m",
    ) == pytest.approx(0.1792, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_inner_volume():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length",
        units="m",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":coolant_flow_length",
        units="m",
        val=0.05,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":highlight_width",
        units="m",
        val=0.1592,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingInnerVolume(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            connected_air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:inner_volume",
        units="m**3",
    ) == pytest.approx(0.0335, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_outer_volume():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length",
        units="m",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:inlet_side_height",
        units="m",
        val=0.0598,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:outlet_side_height",
        units="m",
        val=0.739,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:outlet_side_width",
        units="m",
        val=0.07,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:inlet_side_width",
        units="m",
        val=0.1792,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingOuterVolume(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:outer_volume",
        units="m**3",
    ) == pytest.approx(0.0435, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_weight():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:outer_volume",
        units="m**3",
        val=0.0435,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:inner_volume",
        units="m**3",
        val=0.0335,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingDiffuserWeight(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:mass",
        units="kg",
    ) == pytest.approx(27.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_cross_section_area():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:throat_height",
        units="m",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:highlight_width",
        units="m",
        val=0.1592,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.05,
    )

    problem = run_system(
        SizingCrossSectionArea(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            connected_air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:entrance_area",
        units="m**2",
    ) == pytest.approx(0.00634, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:exit_area",
        units="m**2",
    ) == pytest.approx(0.03595, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_area_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:entrance_area",
        units="m**2",
        val=0.00634,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:exit_area",
        units="m**2",
        val=0.03595,
    )

    problem = run_system(
        SizingAreaRatio(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:area_ratio",
        units="unitless",
    ) == pytest.approx(0.1764, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_stall_check_ratios():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length",
        units="m",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:throat_height",
        units="m",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:highlight_width",
        units="m",
        val=0.1592,
    )

    problem = run_system(
        SizingDiffuserStallCheckRatios(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            connected_air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length_height_ratio",
        units="unitless",
    ) == pytest.approx(25.13, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1"
        ":length_width_ratio",
        units="unitless",
    ) == pytest.approx(6.27, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_entry_hydraulic_diameter():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:throat_height",
        units="m",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:highlight_width",
        units="m",
        val=0.1592,
    )

    problem = run_system(
        SizingEntryHydraulicDiameter(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            connected_air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:entry_hydraulic_diameter",
        units="m",
    ) == pytest.approx(0.0634, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_relative_roughness():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:entry_hydraulic_diameter",
        units="m",
        val=0.0634,
    )

    problem = run_system(
        SizingDiffuserRelativeRoughness(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:relative_roughness",
        units="unitless",
    ) == pytest.approx(0.001577, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_diffuser():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length",
        units="m",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.05,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:throat_height",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:highlight_width",
        units="m",
        val=0.1592,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:wall_thickness",
        units="m",
        val=0.01,
    )

    problem = run_system(
        SizingDiffuser(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            connected_heat_exchanger_id="heat_exchanger_1",
            connected_air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:mass",
        units="kg",
    ) == pytest.approx(27.0, rel=1e-2)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_air_heat_capacity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("throat_air_speed", units="m/s", val=100.0, shape=NB_POINTS_TEST)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:area_ratio",
        units="unitless",
        val=0.5,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesDiffuserExitAirSpeed(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("exit_air_speed", units="m/s") == pytest.approx(
        np.full(NB_POINTS_TEST, 50.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_average_air_speed():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("throat_air_speed", units="m/s", val=100.0, shape=NB_POINTS_TEST)
    ivc.add_output("exit_air_speed", units="m/s", val=50.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesDiffuserAverageAirSpeed(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("average_air_speed", units="m/s") == pytest.approx(
        np.full(NB_POINTS_TEST, 75.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_first_stall_angles():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length_height_ratio",
        units="unitless",
        val=25.13,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length_width_ratio",
        units="unitless",
        val=6.27,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesDiffuserFirstStallAngel(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:first_stall_alpha",
        units="deg",
    ) == pytest.approx(2.07, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:first_stall_beta",
        units="deg",
    ) == pytest.approx(4.88, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_appreciable_stall_angles():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length_height_ratio",
        units="unitless",
        val=25.13,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length_width_ratio",
        units="unitless",
        val=6.27,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesDiffuserAppreciableStallAngles(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:appreciable_stall_alpha",
        units="deg",
    ) == pytest.approx(3.611, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:appreciable_stall_beta",
        units="deg",
    ) == pytest.approx(6.97, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_reynolds_number():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("average_air_speed", units="m/s", val=75.0, shape=NB_POINTS_TEST)
    ivc.add_output("diffuser_air_density", units="kg/m**3", val=1.225, shape=NB_POINTS_TEST)
    ivc.add_output(
        "diffuser_air_dynamic_viscosity", units="Pa*s", val=1.81e-5, shape=NB_POINTS_TEST
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:entry_hydraulic_diameter",
        units="m",
        val=0.0634,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesDiffuserReynoldsNumber(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("air_reynolds_number", units="unitless") == pytest.approx(
        np.full(NB_POINTS_TEST, 321000.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_diffuser_darcy_friction_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("air_reynolds_number", units="unitless", val=321000.0, shape=NB_POINTS_TEST)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:relative_roughness",
        units="unitless",
        val=0.001577,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesDiffuserDarcyFrictionFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("diffuser_darcy_friction_factor", units="unitless") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.0226), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_diffuser_singular_pressure_loss_coeff():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:alpha",
        units="deg",
        val=5.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesDiffuserSingularPressureLossCoefficient(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "diffuser_singular_pressure_loss_coefficient",
        units="unitless",
    ) == pytest.approx(0.19, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_expansion_loss_coeff():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:area_ratio",
        units="unitless",
        val=0.5,
    )
    ivc.add_output(
        "diffuser_singular_pressure_loss_coefficient",
        units="unitless",
        val=0.19,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesDiffuserExpansionLossCoefficient(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "diffuser_expansion_loss_coefficient", units="unitless"
    ) == pytest.approx(0.0475, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_friction_loss_coeff():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:alpha",
        units="deg",
        val=5.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:beta",
        units="deg",
        val=5.0,
    )
    ivc.add_output(
        "diffuser_darcy_friction_factor",
        units="unitless",
        val=0.0226,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:area_ratio",
        units="unitless",
        val=0.5,
    )

    problem = run_system(
        PerformancesDiffuserFrictionLossCoefficient(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("diffuser_friction_loss_coefficient", units="unitless") == pytest.approx(
        0.0243, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_diffuser_pressure_drop():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "diffuser_friction_loss_coefficient", units="unitless", val=0.0243, shape=NB_POINTS_TEST
    )
    ivc.add_output("diffuser_expansion_loss_coefficient", units="unitless", val=0.0475)
    ivc.add_output("diffuser_air_density", units="kg/m**3", val=1.225, shape=NB_POINTS_TEST)
    ivc.add_output("average_air_speed", units="m/s", val=75.0, shape=NB_POINTS_TEST)

    problem = run_system(
        PerformancesDiffuserPressureDrop(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:air_pressure_drop",
        units="Pa",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 247.37), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_exit_total_pressure():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:air_pressure_drop",
        units="Pa",
        val=247.37,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "throat_air_pressure",
        units="Pa",
        val=101325.0,
        shape=NB_POINTS_TEST,
    )

    problem = run_system(
        PerformancesDiffuserExitTotalPressure(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "diffuser_exit_total_pressure",
        units="Pa",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 101077.63), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_exit_total_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("exit_air_speed", units="m/s", val=50.0, shape=NB_POINTS_TEST)
    ivc.add_output(
        "diffuser_air_specific_heat_capacity", units="J/kg/K", val=1005.0, shape=NB_POINTS_TEST
    )
    ivc.add_output("throat_air_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("diffuser_air_density", units="kg/m**3", val=1.225, shape=NB_POINTS_TEST)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1"
        ":air_pressure_drop",
        units="Pa",
        val=247.37,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output("throat_air_speed", units="m/s", val=100.0, shape=NB_POINTS_TEST)

    problem = run_system(
        PerformancesDiffuserExitTotalTemperature(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "diffuser_exit_total_temperature",
        units="K",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 303.5), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diffuser_performance():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("throat_air_speed", units="m/s", val=100.0, shape=NB_POINTS_TEST)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1" ":area_ratio",
        units="unitless",
        val=0.5,
    )
    ivc.add_output("throat_air_pressure", units="Pa", val=101325.0, shape=NB_POINTS_TEST)
    ivc.add_output("throat_air_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:alpha",
        units="deg",
        val=5.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:beta",
        units="deg",
        val=5.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:entry_hydraulic_diameter",
        units="m",
        val=0.0634,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:relative_roughness",
        units="unitless",
        val=0.001577,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1"
        ":length_height_ratio",
        units="unitless",
        val=25.13,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:diffuser_1:length_width_ratio",
        units="unitless",
        val=6.27,
    )

    problem = run_system(
        PerformancesDiffuser(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            diffuser_id="diffuser_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "diffuser_exit_total_temperature",
        units="K",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 303.5), rel=1e-2)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))
