# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import os.path as pth
import pytest
import numpy as np
import openmdao.api as om

from ..perf_max_inlet_boundary_layer_thickness import PerformancesMaxBoundaryLayerThickness
from ..perf_throat_height_momentum_layer_thickness_ratio import (
    PerformancesThroatHeightMomentumBoundaryLayerThicknessRatio,
)
from ..perf_momentum_flow_correction_factor import PerformancesMomentumFlowCorrectionFactor
from ..perf_boundary_layer_thickness_highlight_height_ratio import (
    PerformancesBoundaryLayerThicknessHighlightHeightRatio,
)
from ..perf_modified_mass_flow_ratio import PerformancesModifiedMassFlowRatio
from ..perf_air_mass_flow_ratio import PerformancesAirMassFlowRatio
from ..perf_drag_correlation_factor import PerformancesDragCorrelationFactor
from ..perf_drag_ksp_factor import PerformancesDragKspFactor
from ..perf_ramp_angle_factor import PerformancesRampAngleFactor
from ..perf_mach_factor import PerformancesMachFactor
from ..perf_drag_coefficient_zero import PerformancesCDZeroInletMassFlow
from ..perf_inlet_drag import _PerformancesInletDrag, PerformancesInletDrag
from ..perf_air_dynamic_pressure import PerformancesAirDynamicPressure
from ..perf_max_ramp_pressure_efficiency import PerformancesMaxRamPressureEfficiency
from ..perf_pressure_efficiency_difference_factor import (
    PerformancesPressureEfficiencyDifferenceFactor,
)
from ..perf_pressure_efficiency_difference import PerformancesPressureEfficiencyDifference
from ..perf_inlet_efficiency import PerformancesInletEfficiency
from ..perf_throat_airspeed import PerformancesThroatAirSpeed
from ..perf_ambient_total_pressure import PerformancesAmbientTotalPressure
from ..perf_throat_total_temperature import PerformancesThroatTemperature
from ..perf_throat_total_pressure import PerformancesThroatPressure
from ..perf_inlet_pressure_drop import PerformancesInletPressureDrop
from ..perf_inlet import PerformancesInlet

from ..sizing_throat_height import SizingThroatHeight
from ..sizing_inlet_geometry import SizingInletGeometry
from ..sizing_inlet_weight import SizingInletWeight

from tests.testing_utilities import run_system

NB_POINTS_TEST = 10


def test_inlet_air_dynamic_pressure():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.full(NB_POINTS_TEST, 108.0),
    )
    ivc.add_output(
        "density",
        units="kg/m**3",
        val=np.full(NB_POINTS_TEST, 1.225),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAirDynamicPressure(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("dynamic_pressure", units="Pa") == pytest.approx(
        np.full(NB_POINTS_TEST, 7144.2), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_inlet_max_boundary_layer_thickness():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "dynamic_viscosity",
        units="Pa*s",
        val=np.full(NB_POINTS_TEST, 1.79e-5),
    )
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.full(NB_POINTS_TEST, 108.0),
    )
    ivc.add_output(
        "density",
        units="kg/m**3",
        val=np.full(NB_POINTS_TEST, 1.225),
    )
    ivc.add_output(
        "mach",
        val=np.full(NB_POINTS_TEST, 0.33),
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:ramp_length",
        units="m",
        val=1.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaxBoundaryLayerThickness(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            number_of_points=NB_POINTS_TEST,
            air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":max_boundary_layer_thickness",
        units="m",
    ) == pytest.approx(0.0158, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":max_momentum_boundary_layer_thickness",
        units="m",
    ) == pytest.approx(0.00158, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":design_air_density",
        units="kg/m**3",
    ) == pytest.approx(1.225, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":design_true_airspeed",
        units="m/s",
    ) == pytest.approx(108.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":design_dynamic_viscosity",
        units="Pa*s",
    ) == pytest.approx(1.79e-5, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:design_mach",
    ) == pytest.approx(0.33, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_throat_height_momentum_thickness_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:max_momentum_boundary_layer_thickness",
        units="m",
        val=0.00158,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:design_true_airspeed",
        units="m/s",
        val=108.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:design_air_density",
        units="kg/m**3",
        val=1.225,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:design_air_mass_flow",
        units="kg/s",
        val=0.5,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesThroatHeightMomentumBoundaryLayerThicknessRatio(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height_layer_thickness_ratio",
    ) == pytest.approx(0.0398, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_throat_height():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height_layer_thickness_ratio",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":max_momentum_boundary_layer_thickness",
        units="m",
        val=0.00158,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingThroatHeight(pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        units="m",
    ) == pytest.approx(0.0398, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_geometry():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":ramp_angle",
        val=7.0,
        units="deg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingInletGeometry(pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":highlight_width",
        units="m",
    ) == pytest.approx(0.1592, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_lip_thickness",
        units="m",
    ) == pytest.approx(0.00995, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":lip_length",
        units="m",
    ) == pytest.approx(0.00995, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":ramp_floor_inlet_plane_distance",
        units="m",
    ) == pytest.approx(0.0485, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":lip_ramp_floor_distance",
        units="m",
    ) == pytest.approx(0.0436, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":inlet_capture_area",
        units="ft**2",
    ) == pytest.approx(0.00693, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_boundary_layer_thickness_highlight_height_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":max_boundary_layer_thickness",
        units="m",
        val=0.0158,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":lip_ramp_floor_distance",
        units="m",
        val=0.0436,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesBoundaryLayerThicknessHighlightHeightRatio(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":layer_thickness_highlight_height_ratio",
    ) == pytest.approx(0.362, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_momentum_flow_correction_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":layer_thickness_highlight_height_ratio",
        val=0.362,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":design_mach",
        val=0.33,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMomentumFlowCorrectionFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        ivc,
    )

    assert problem.get_val("momentum_flow_correction_factor") == pytest.approx(0.93, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_modified_mass_flow_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height_layer_thickness_ratio",
        val=0.0398,
    )

    problem = run_system(
        PerformancesModifiedMassFlowRatio(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        ivc,
    )

    assert problem.get_val("modified_mass_flow_ratio") == pytest.approx(0.613, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_air_mass_flow_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        units="m",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":lip_ramp_floor_distance",
        units="m",
        val=0.0436,
    )
    ivc.add_output(
        "modified_mass_flow_ratio",
        val=0.613,
    )

    problem = run_system(
        PerformancesAirMassFlowRatio(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        ivc,
    )

    assert problem.get_val("air_mass_flow_ratio") == pytest.approx(0.56, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_drag_correlation_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("air_mass_flow_ratio", val=0.56)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":design_mach",
        val=0.33,
    )

    problem = om.Problem(reports=False)
    model = problem.model
    model.add_subsystem(
        name="ivc",
        subsys=ivc,
        promotes=["*"],
    )
    model.add_subsystem(
        name="dc_current",
        subsys=PerformancesDragCorrelationFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        promotes=["*"],
    )
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    model.nonlinear_solver.options["iprint"] = 0
    model.nonlinear_solver.options["maxiter"] = 200
    model.nonlinear_solver.options["rtol"] = 1e-5
    model.linear_solver = om.DirectSolver()

    problem.setup()
    problem.run_model()

    assert problem.get_val("drag_correlation_factor") == pytest.approx(-0.022, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_drag_ksp_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":design_mach",
        val=0.4,
    )
    ivc.add_output(
        "air_mass_flow_ratio",
        val=0.4,
    )

    problem = run_system(
        PerformancesDragKspFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        ivc,
    )

    assert problem.get_val("k_sp_factor") == pytest.approx(-0.03, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_ramp_angle_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":ramp_angle",
        val=8.0,
        units="deg",
    )

    problem = run_system(
        PerformancesRampAngleFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        ivc,
    )

    assert problem.get_val("ramp_angle_factor") == pytest.approx(1.12, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_mach_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":design_mach",
        val=0.55,
    )

    problem = run_system(
        PerformancesMachFactor(pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"),
        ivc,
    )

    assert problem.get_val("mach_factor") == pytest.approx(1.09, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_cd0():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":lip_length",
        units="m",
        val=0.00995,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":lip_ramp_floor_distance",
        units="m",
        val=0.0436,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":ramp_floor_inlet_plane_distance",
        units="m",
        val=0.0485,
    )

    problem = run_system(
        PerformancesCDZeroInletMassFlow(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        ivc,
    )

    assert problem.get_val("cd_zero_inlet_mass_flow") == pytest.approx(0.16, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_drag():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.full(NB_POINTS_TEST, 108.0),
    )
    ivc.add_output("mach_factor", val=1.09)
    ivc.add_output("ramp_angle_factor", val=1.12)
    ivc.add_output("k_sp_factor", val=-0.03)
    ivc.add_output("drag_correlation_factor", val=-0.022)
    ivc.add_output("cd_zero_inlet_mass_flow", val=0.16)
    ivc.add_output("air_mass_flow_ratio", val=0.56)
    ivc.add_output("momentum_flow_correction_factor", val=0.93)
    ivc.add_output(
        "air_mass_flow",
        units="kg/s",
        val=np.full(NB_POINTS_TEST, 0.5),
    )

    problem = run_system(
        _PerformancesInletDrag(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            number_of_points=NB_POINTS_TEST,
            air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:drag",
        units="N",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 48.88), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_drag_group():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "altitude",
        units="m",
        val=np.full(NB_POINTS_TEST, 0.0),
    )
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=108.0,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "density",
        units="kg/m**3",
        val=np.full(NB_POINTS_TEST, 1.225),
    )
    ivc.add_output(
        "mach",
        val=np.full(NB_POINTS_TEST, 0.33),
    )
    ivc.add_output(
        "air_mass_flow",
        units="kg/s",
        val=np.full(NB_POINTS_TEST, 0.5),
    )
    ivc.add_output("dynamic_viscosity", units="Pa*s", val=np.full(NB_POINTS_TEST, 1.79e-5))
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":design_air_mass_flow",
        units="kg/s",
        val=0.5,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        units="m",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":lip_ramp_floor_distance",
        units="m",
        val=0.0436,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":ramp_floor_inlet_plane_distance",
        units="m",
        val=0.0485,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":lip_length",
        units="m",
        val=0.00995,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesInletDrag(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            air_inlet_id="air_inlet_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:drag",
        units="N",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 49.08), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_max_ram_pressure_efficiency():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height_layer_thickness_ratio",
        val=0.0398,
    )

    problem = run_system(
        PerformancesMaxRamPressureEfficiency(
            pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":max_ram_pressure_efficiency"
    ) == pytest.approx(0.9, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_pressure_efficiency_difference_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.full(NB_POINTS_TEST, 108.0),
    )
    ivc.add_output(
        "density",
        units="kg/m**3",
        val=np.full(NB_POINTS_TEST, 1.225),
    )
    ivc.add_output("air_mass_flow", units="kg/s", val=np.full(NB_POINTS_TEST, 0.5))
    ivc.add_output("dynamic_viscosity", units="Pa*s", val=np.full(NB_POINTS_TEST, 1.79e-5))
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        units="m",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":highlight_width",
        units="m",
        val=0.1592,
    )

    problem = run_system(
        PerformancesPressureEfficiencyDifferenceFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            air_inlet_id="air_inlet_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("pressure_efficiency_difference_factor") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.596), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_inlet_pressure_efficiency_difference():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "pressure_efficiency_difference_factor",
        val=np.full(NB_POINTS_TEST, 0.596),
    )

    problem = run_system(
        PerformancesPressureEfficiencyDifference(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("pressure_efficiency_difference") == pytest.approx(
        np.full(NB_POINTS_TEST, -0.054), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_inlet_efficiency():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":max_ram_pressure_efficiency",
        val=0.9,
    )
    ivc.add_output(
        "pressure_efficiency_difference",
        val=np.full(NB_POINTS_TEST, -0.054),
    )

    problem = run_system(
        PerformancesInletEfficiency(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            air_inlet_id="air_inlet_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("inlet_efficiency") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.846), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_inlet_throat_airspeed():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "density",
        units="kg/m**3",
        val=np.full(NB_POINTS_TEST, 1.225),
    )
    ivc.add_output(
        "air_mass_flow",
        units="kg/s",
        val=np.full(NB_POINTS_TEST, 0.5),
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        units="m",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:highlight_width",
        units="m",
        val=0.1592,
    )

    problem = run_system(
        PerformancesThroatAirSpeed(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            number_of_points=NB_POINTS_TEST,
            air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "throat_air_speed",
        units="m/s",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 64.4), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_ambient_total_pressure():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "mach",
        val=np.full(NB_POINTS_TEST, 0.33),
    )
    ivc.add_output(
        "ambient_pressure",
        units="Pa",
        val=np.full(NB_POINTS_TEST, 101325.0),
    )

    problem = run_system(
        PerformancesAmbientTotalPressure(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            number_of_points=NB_POINTS_TEST,
            air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "ambient_total_pressure",
        units="Pa",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 109261.6), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_ambient_total_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "mach",
        val=np.full(NB_POINTS_TEST, 0.33),
    )
    ivc.add_output(
        "exterior_temperature",
        units="K",
        val=np.full(NB_POINTS_TEST, 288.15),
    )

    problem = run_system(
        PerformancesThroatTemperature(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            number_of_points=NB_POINTS_TEST,
            air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "throat_total_temperature",
        units="K",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 294.4), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_throat_total_pressure():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "ambient_total_pressure",
        units="Pa",
        val=np.full(NB_POINTS_TEST, 109261.6),
    )
    ivc.add_output(
        "ambient_pressure",
        units="Pa",
        val=np.full(NB_POINTS_TEST, 101325.0),
    )
    ivc.add_output(
        "inlet_efficiency",
        val=np.full(NB_POINTS_TEST, 0.846),
    )

    problem = run_system(
        PerformancesThroatPressure(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val(
        "throat_total_pressure",
        units="Pa",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 108039.36), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_max_pressure_drop():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("dynamic_pressure", units="Pa", val=np.full(NB_POINTS_TEST, 7144.2))
    ivc.add_output("throat_total_pressure", units="Pa", val=np.full(NB_POINTS_TEST, 108039.36))

    problem = run_system(
        PerformancesInletPressureDrop(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            number_of_points=NB_POINTS_TEST,
            air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1"
        ":air_inlet_1:air_pressure_drop",
        units="Pa",
    ) == pytest.approx(100895.16, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:"
        "max_ambient_dynamic_pressure",
        units="Pa",
    ) == pytest.approx(7144.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_performance():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("attitude", units="m", val=np.full(NB_POINTS_TEST, 0.0))
    ivc.add_output("true_airspeed", units="m/s", val=np.full(NB_POINTS_TEST, 108.0))
    ivc.add_output("density", units="kg/m**3", val=np.full(NB_POINTS_TEST, 1.225))
    ivc.add_output("mach", val=np.full(NB_POINTS_TEST, 0.33))
    ivc.add_output(
        "air_mass_flow",
        units="kg/s",
        val=np.full(NB_POINTS_TEST, 0.5),
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        units="m",
        val=0.0398,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":highlight_width",
        units="m",
        val=0.1592,
    )
    ivc.add_output(
        "exterior_temperature",
        units="K",
        val=np.full(NB_POINTS_TEST, 288.15),
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:lip_ramp_floor_distance",
        units="m",
        val=0.0436,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:lip_length",
        units="m",
        val=0.00995,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:ramp_floor_inlet_plane_distance",
        units="m",
        val=0.0485,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:ramp_length",
        units="m",
        val=1.0,
    )

    problem = run_system(
        PerformancesInlet(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            air_inlet_id="air_inlet_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("inlet_efficiency") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.846), rel=1e-2
    )
    assert problem.get_val("throat_total_pressure", units="Pa") == pytest.approx(
        np.full(NB_POINTS_TEST, 108039.36), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_inlet_mass():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:ramp_length",
        units="m",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:inlet_capture_area",
        units="m**2",
        val=0.000643,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:max_ambient_dynamic_pressure",
        units="Pa",
        val=7144.2,
    )

    problem = run_system(
        SizingInletWeight(pemfc_stack_bop_id="pemfc_stack_bop_1", air_inlet_id="air_inlet_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:mass",
        units="kg",
    ) == pytest.approx(0.397, rel=1e-2)

    problem.check_partials(compact_print=True)
