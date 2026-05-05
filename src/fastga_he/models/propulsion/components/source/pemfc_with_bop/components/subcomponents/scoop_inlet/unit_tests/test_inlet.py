# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import os.path as pth
import pytest
import numpy as np
import openmdao.api as om


from ..perf_air_dynamic_pressure import PerformancesAirDynamicPressure
from ..perf_throat_airspeed import PerformancesThroatAirSpeed
from ..perf_ambient_total_pressure import PerformancesAmbientTotalPressure
from ..perf_scoop_inlet import PerformancesScoopInlet
from ..perf_design_flow_area import PerformancesInletDesignFlowArea
from ..perf_inlet_design_dynamic_pressure import PerformancesInletDesignDynamicPressure

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


def test_inlet_design_flow_area():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "total_air_mass_flow",
        units="kg/s",
        val=np.full(NB_POINTS_TEST, 0.05),
    )
    ivc.add_output(
        "throat_air_speed",
        units="m/s",
        val=np.full(NB_POINTS_TEST, 90.0),
    )
    ivc.add_output(
        "density",
        units="kg/m**3",
        val=np.full(NB_POINTS_TEST, 1.225),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesInletDesignFlowArea(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            air_inlet_id="air_inlet_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:design_flow_area",
        units="m**2",
    ) == pytest.approx(0.00045, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_geometry():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:design_flow_area",
        units="m**2",
        val=0.00045,
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
    ) == pytest.approx(0.0424, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":throat_height",
        units="m",
    ) == pytest.approx(0.0106, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1"
        ":inlet_capture_area",
        units="ft**2",
    ) == pytest.approx(0.00484, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_throat_airspeed():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.full(NB_POINTS_TEST, 90.0),
    )

    problem = run_system(
        PerformancesThroatAirSpeed(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "throat_air_speed",
        units="m/s",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 90.0), rel=1e-2)

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


def test_inlet_design_dynamic_pressure():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "dynamic_pressure",
        units="Pa",
        val=np.full(NB_POINTS_TEST, 7144.2),
    )

    problem = run_system(
        PerformancesInletDesignDynamicPressure(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            air_inlet_id="air_inlet_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:"
        "design_ambient_dynamic_pressure",
        units="Pa",
    ) == pytest.approx(7144.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inlet_performance():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("altitude", units="m", val=np.full(NB_POINTS_TEST, 0.0))
    ivc.add_output("true_airspeed", units="m/s", val=np.full(NB_POINTS_TEST, 108.0))
    ivc.add_output("density", units="kg/m**3", val=np.full(NB_POINTS_TEST, 1.225))
    ivc.add_output("mach", val=np.full(NB_POINTS_TEST, 0.33))
    ivc.add_output(
        "air_consumption",
        units="kg/s",
        val=np.full(NB_POINTS_TEST, 0.05),
    )

    problem = run_system(
        PerformancesScoopInlet(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            air_inlet_id="air_inlet_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_inlet_mass():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:length",
        units="m",
        val=0.5,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:inlet_capture_area",
        units="m**2",
        val=0.0045,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_inlet_1:design_ambient_dynamic_pressure",
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
    ) == pytest.approx(0.498, rel=1e-2)

    problem.check_partials(compact_print=True)
