# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_fuel_system_cg_x import SizingFuelSystemCGX
from ..components.sizing_fuel_system_cg_y import SizingFuelSystemCGY
from ..components.sizing_fuel_system_volume import SizingFuelSystemCapacityVolume
from ..components.sizing_fuel_system_weight import SizingFuelSystemWeight

from ..components.pre_lca_prod_weight_per_fu import PreLCAFuelSystemProdWeightPerFU

from ..components.perf_fuel_output import PerformancesFuelOutput
from ..components.perf_fuel_input import PerformancesFuelInput
from ..components.perf_total_fuel_flowed import PerformancesTotalFuelFlowed

from ..components.sizing_fuel_system import SizingFuelSystem
from ..components.perf_fuel_system import PerformancesFuelSystem

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_hydrogen_pipeline.xml"
NB_POINTS_TEST = 10


def test_fuel_system_cg_x():
    expected_cg = [2.69, 0.5, 1.98]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingFuelSystemCGX(fuel_system_id="fuel_system_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingFuelSystemCGX(fuel_system_id="fuel_system_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:fuel_system:fuel_system_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_fuel_system_cg_y():
    expected_cg = [1.344, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingFuelSystemCGY(fuel_system_id="fuel_system_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingFuelSystemCGY(fuel_system_id="fuel_system_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:fuel_system:fuel_system_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_fuel_system_volume():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingFuelSystemCapacityVolume(fuel_system_id="fuel_system_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingFuelSystemCapacityVolume(fuel_system_id="fuel_system_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:connected_volume", units="L"
    ) == pytest.approx(62.59, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuel_system_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingFuelSystemWeight(fuel_system_id="fuel_system_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingFuelSystemWeight(fuel_system_id="fuel_system_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:mass", units="kg"
    ) == pytest.approx(10.98, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuel_system_weight_jet_fuel():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:he_power_train:fuel_system:fuel_system_1:fuel_type", val=3.0)
    ivc.add_output(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:number_engine", val=1.0
    )
    ivc.add_output(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:connected_volume",
        val=62.59,
        units="L",
    )

    problem = run_system(SizingFuelSystemWeight(fuel_system_id="fuel_system_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:mass", units="kg"
    ) == pytest.approx(3.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuel_output():
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_out_t_1", val=np.linspace(1.0, 2.0, NB_POINTS_TEST), units="kg")
    ivc.add_output("fuel_consumed_out_t_2", val=np.linspace(4.0, 2.0, NB_POINTS_TEST), units="kg")

    problem = run_system(
        PerformancesFuelOutput(
            fuel_system_id="fuel_system_1", number_of_points=NB_POINTS_TEST, number_of_engines=2
        ),
        ivc,
    )

    assert problem.get_val("fuel_flowing_t", units="kg") == pytest.approx(
        np.linspace(5.0, 4.0, NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:number_engine"
    ) == pytest.approx(2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuel_input():
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_flowing_t", val=np.linspace(5.0, 4.0, NB_POINTS_TEST), units="kg")

    problem = run_system(
        PerformancesFuelInput(
            fuel_system_id="fuel_system_1", number_of_points=NB_POINTS_TEST, number_of_tanks=2
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
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:fuel_distribution",
        val=np.array([3.0 / 4.0, 1.0 / 4.0]),
    )

    problem = run_system(
        PerformancesFuelInput(
            fuel_system_id="fuel_system_1", number_of_points=NB_POINTS_TEST, number_of_tanks=2
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
        PerformancesTotalFuelFlowed(
            fuel_system_id="fuel_system_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:total_fuel_flowed", units="kg"
    ) == pytest.approx(45.0, rel=1e-2)


def test_fuel_system_performances():
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_out_t_1", val=np.linspace(1.0, 2.0, NB_POINTS_TEST), units="kg")
    ivc.add_output("fuel_consumed_out_t_2", val=np.linspace(4.0, 2.0, NB_POINTS_TEST), units="kg")

    problem = run_system(
        PerformancesFuelSystem(
            number_of_points=NB_POINTS_TEST,
            number_of_engines=2,
            number_of_tanks=5,
            fuel_system_id="fuel_system_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:total_fuel_flowed", units="kg"
    ) == pytest.approx(45.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:number_engine"
    ) == pytest.approx(2, rel=1e-2)
    assert problem.get_val("fuel_consumed_in_t_1", units="kg") == pytest.approx(
        np.linspace(1.0, 0.8, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_sizing_tank():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingFuelSystem(fuel_system_id="fuel_system_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingFuelSystem(fuel_system_id="fuel_system_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:mass", units="kg"
    ) == pytest.approx(10.98, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:low_speed:CD0"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:mass",
        "data:environmental_impact:aircraft_per_fu",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCAFuelSystemProdWeightPerFU(fuel_system_id="fuel_system_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_system:fuel_system_1:mass_per_fu", units="kg"
    ) == pytest.approx(1.098e-05, rel=1e-3)

    problem.check_partials(compact_print=True)
