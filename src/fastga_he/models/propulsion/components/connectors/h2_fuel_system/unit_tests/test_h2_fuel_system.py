# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import pytest
import numpy as np
import os.path as pth
import openmdao.api as om

from ..components.sizing_h2_fuel_system_cg_x import SizingH2FuelSystemCGX
from ..components.sizing_h2_fuel_system_cg_y import SizingH2FuelSystemCGY
from ..components.sizing_h2_fuel_system_length import SizingH2FuelSystemLength
from ..components.sizing_h2_fuel_system_weight import SizingH2FuelSystemWeight
from ..components.sizing_h2_fuel_system_inner_diameter import SizingH2FuelSystemInnerDiameter
from ..components.sizing_h2_fuel_system_relative_roughness import (
    SizingH2FuelSystemRelativeRoughness,
)
from ..components.sizing_h2_fuel_system_cross_section import (
    SizingH2FuelSystemCrossSectionDimension,
)
from ..components.sizing_h2_fuel_system import SizingH2FuelSystem

from ..components.perf_h2_fuel_output import PerformancesH2FuelSystemOutput
from ..components.perf_h2_fuel_input import PerformancesH2FuelSystemInput
from ..components.perf_h2_total_fuel_flowed import PerformancesTotalH2FuelFlowed
from ..components.perf_h2_fuel_system import PerformancesH2FuelSystem

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_hydrogen_pipeline.xml"
NB_POINTS_TEST = 10


def test_fuel_system_cg_x():
    expected_cg = [1.869, 2.239, 2.608]
    expected_cg_wing = [2.054, 2.239, 2.423]

    for option, expected_value, expected_value_wing in zip(
        POSSIBLE_POSITION, expected_cg, expected_cg_wing
    ):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingH2FuelSystemCGX(h2_fuel_system_id="h2_fuel_system_1", position=option)
            ),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingH2FuelSystemCGX(h2_fuel_system_id="h2_fuel_system_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        ivc = get_indep_var_comp(
            list_inputs(
                SizingH2FuelSystemCGX(
                    h2_fuel_system_id="h2_fuel_system_1", position=option, wing_related=True
                )
            ),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingH2FuelSystemCGX(
                h2_fuel_system_id="h2_fuel_system_1", position=option, wing_related=True
            ),
            ivc,
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:CG:x", units="m"
        ) == pytest.approx(expected_value_wing, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_fuel_system_cg_y():
    expected_cg = [0.672, 0.672, 0.672]
    expected_cg_compact = [1.344, 1.344, 1.344]

    for option, expected_value, expected_value_compact in zip(
        POSSIBLE_POSITION, expected_cg, expected_cg_compact
    ):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingH2FuelSystemCGY(h2_fuel_system_id="h2_fuel_system_1", wing_related=True)
            ),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingH2FuelSystemCGY(h2_fuel_system_id="h2_fuel_system_1"), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:CG:y", units="m"
        ) == pytest.approx(0.0, rel=1e-2)

        problem = run_system(
            SizingH2FuelSystemCGY(h2_fuel_system_id="h2_fuel_system_1", wing_related=True), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)

        problem = run_system(
            SizingH2FuelSystemCGY(
                h2_fuel_system_id="h2_fuel_system_1", wing_related=True, compact=True
            ),
            ivc,
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:CG:y", units="m"
        ) == pytest.approx(expected_value_compact, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_h2_fuel_pipe_length():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingH2FuelSystemLength(h2_fuel_system_id="h2_fuel_system_1", compact=True)),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingH2FuelSystemLength(h2_fuel_system_id="h2_fuel_system_1", compact=True), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:length", units="m"
    ) == pytest.approx(2.0, rel=1e-2)

    problem.check_partials(compact_print=True)

    expected_length = [0.7387, 1.4774, 0.7387]
    expected_length_wing = [2.083, 1.4774, 2.083]

    for position, expected_value, expected_value_wing in zip(
        POSSIBLE_POSITION, expected_length, expected_length_wing
    ):
        ivc = get_indep_var_comp(
            list_inputs(
                SizingH2FuelSystemLength(h2_fuel_system_id="h2_fuel_system_1", position=position)
            ),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingH2FuelSystemLength(h2_fuel_system_id="h2_fuel_system_1", position=position), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:length",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        ivc = get_indep_var_comp(
            list_inputs(
                SizingH2FuelSystemLength(
                    h2_fuel_system_id="h2_fuel_system_1", position=position, wing_related=True
                )
            ),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingH2FuelSystemLength(
                h2_fuel_system_id="h2_fuel_system_1", position=position, wing_related=True
            ),
            ivc,
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:length",
            units="m",
        ) == pytest.approx(expected_value_wing, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_h2_fuel_inner_diameter():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:pipe_pressure",
        val=70.0,
        units="MPa",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:pipe_diameter",
        val=0.04,
        units="m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:material:yield_strength",
        val=240,
        units="MPa",
    )

    problem = run_system(SizingH2FuelSystemInnerDiameter(h2_fuel_system_id="h2_fuel_system_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:inner_diameter",
        units="m",
    ) == pytest.approx(0.0231, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_h2_fuel_inner_diameter_clipped():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:pipe_pressure" "",
        val=1013250.0,
        units="Pa",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:pipe_diameter",
        val=0.04,
        units="m",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:material" ":yield_strength",
        val=240,
        units="MPa",
    )

    problem = run_system(SizingH2FuelSystemInnerDiameter(h2_fuel_system_id="h2_fuel_system_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:inner_diameter",
        units="m",
    ) == pytest.approx(0.037, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_h2_fuel_relative_roughness():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:inner_diameter",
        units="m",
        val=0.0373,
    )

    problem = run_system(
        SizingH2FuelSystemRelativeRoughness(h2_fuel_system_id="h2_fuel_system_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:relative_roughness",
    ) == pytest.approx(0.001206, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_h2_fuel_overall_cross_section():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:inner_diameter",
        units="m",
        val=0.0373,
    )

    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:pipe_diameter",
        units="m",
        val=0.04,
    )

    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:insulation_thickness",
        units="m",
        val=0.005,
    )

    problem = run_system(
        SizingH2FuelSystemCrossSectionDimension(h2_fuel_system_id="h2_fuel_system_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:overall_diameter",
        units="m",
    ) == pytest.approx(0.05, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:dimension:overall_wall_thickness",
        units="m",
    ) == pytest.approx(0.00635, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuel_system_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingH2FuelSystemWeight(h2_fuel_system_id="h2_fuel_system_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingH2FuelSystemWeight(h2_fuel_system_id="h2_fuel_system_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:mass", units="kg"
    ) == pytest.approx(2.75, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuel_output():
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_out_t_1", val=np.linspace(1.0, 2.0, NB_POINTS_TEST), units="kg")
    ivc.add_output("fuel_consumed_out_t_2", val=np.linspace(4.0, 2.0, NB_POINTS_TEST), units="kg")

    problem = run_system(
        PerformancesH2FuelSystemOutput(
            h2_fuel_system_id="h2_fuel_system_1",
            number_of_points=NB_POINTS_TEST,
            number_of_power_sources=2,
        ),
        ivc,
    )

    assert problem.get_val("fuel_flowing_t", units="kg") == pytest.approx(
        np.linspace(5.0, 4.0, NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:number_source"
    ) == pytest.approx(2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuel_input():
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_flowing_t", val=np.linspace(5.0, 4.0, NB_POINTS_TEST), units="kg")
    problem = run_system(
        PerformancesH2FuelSystemInput(
            h2_fuel_system_id="h2_fuel_system_1",
            number_of_points=NB_POINTS_TEST,
            number_of_tanks=2,
        ),
        ivc,
    )

    assert problem.get_val("fuel_consumed_in_t_1", units="kg") == pytest.approx(
        np.linspace(2.5, 2.0, NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val("fuel_consumed_in_t_2", units="kg") == pytest.approx(
        np.linspace(2.5, 2.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc.add_output(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:fuel_distribution",
        val=np.array([3.0 / 4.0, 1.0 / 4.0]),
    )

    problem = run_system(
        PerformancesH2FuelSystemInput(
            h2_fuel_system_id="h2_fuel_system_1",
            number_of_points=NB_POINTS_TEST,
            number_of_tanks=2,
        ),
        ivc,
    )

    assert problem.get_val("fuel_consumed_in_t_1", units="kg") == pytest.approx(
        np.linspace(3.75, 3.0, NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val("fuel_consumed_in_t_2", units="kg") == pytest.approx(
        np.linspace(1.25, 1.0, NB_POINTS_TEST), rel=1e-2
    )


def test_total_fuel_flowed():
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_flowing_t", val=np.linspace(5.0, 4.0, NB_POINTS_TEST), units="kg")

    problem = run_system(
        PerformancesTotalH2FuelFlowed(
            h2_fuel_system_id="h2_fuel_system_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:total_fuel_flowed",
        units="kg",
    ) == pytest.approx(45.0, rel=1e-2)


def test_fuel_system_performances():
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_out_t_1", val=np.linspace(1.0, 2.0, NB_POINTS_TEST), units="kg")
    ivc.add_output("fuel_consumed_out_t_2", val=np.linspace(4.0, 2.0, NB_POINTS_TEST), units="kg")

    problem = run_system(
        PerformancesH2FuelSystem(
            number_of_points=NB_POINTS_TEST,
            number_of_power_sources=2,
            number_of_tanks=5,
            h2_fuel_system_id="h2_fuel_system_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:total_fuel_flowed",
        units="kg",
    ) == pytest.approx(45.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:number_source"
    ) == pytest.approx(2, rel=1e-2)
    assert problem.get_val("fuel_consumed_in_t_1", units="kg") == pytest.approx(
        np.linspace(1.0, 0.8, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)
    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_sizing_h2_fuel_system():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingH2FuelSystem(h2_fuel_system_id="h2_fuel_system_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingH2FuelSystem(h2_fuel_system_id="h2_fuel_system_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:mass", units="kg"
    ) == pytest.approx(2.23, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:H2_fuel_system:h2_fuel_system_1:low_speed:CD0"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)
    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))
