import pytest
import os.path as pth

from ..rta_propulsion_weight import RTAPropulsionWeight
from ..rta_aero_approximation import (
    ClRef,
    InducedDragCoefficient,
    AeroApproximation,
)
from ..compute_simple_rta_variables import ComputeRTAVariable

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")

XML_FILE = "data.xml"


def test_rta_propulsion_weight():
    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", "fuel_propulsion.yml")
    ivc = get_indep_var_comp(
        list_inputs(RTAPropulsionWeight(power_train_file_path=sample_power_train_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        RTAPropulsionWeight(power_train_file_path=sample_power_train_file_path), ivc
    )

    assert problem.get_val("data:weight:propulsion:engine:CG:x", units="m") == pytest.approx(
        9.355,
        rel=1e-3,
    )
    assert problem.get_val("data:weight:propulsion:engine:mass", units="kg") == pytest.approx(
        992.66,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", "hybrid_propulsion.yml")
    ivc = get_indep_var_comp(
        list_inputs(RTAPropulsionWeight(power_train_file_path=sample_power_train_file_path)),
        __file__,
        "data_hybrid.xml",
    )

    problem = run_system(
        RTAPropulsionWeight(power_train_file_path=sample_power_train_file_path), ivc
    )

    assert problem.get_val("data:weight:propulsion:engine:CG:x", units="m") == pytest.approx(
        10.367,
        rel=1e-3,
    )
    assert problem.get_val("data:weight:propulsion:engine:mass", units="kg") == pytest.approx(
        600.0,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_cl_ref():
    ivc = get_indep_var_comp(list_inputs(ClRef()), __file__, XML_FILE)

    problem = run_system(ClRef(), ivc)

    assert problem.get_val("data:aerodynamics:wing:low_speed:CL_ref") == pytest.approx(
        0.708,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_induced_drag():
    ivc = get_indep_var_comp(list_inputs(InducedDragCoefficient()), __file__, XML_FILE)

    problem = run_system(InducedDragCoefficient(), ivc)

    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
    ) == pytest.approx(
        0.08234,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_aero_approx():
    ivc = get_indep_var_comp(list_inputs(AeroApproximation()), __file__, XML_FILE)

    problem = run_system(AeroApproximation(), ivc)

    assert problem.get_val("data:aerodynamics:wing:low_speed:CL_ref") == pytest.approx(
        0.708,
        rel=1e-3,
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
    ) == pytest.approx(
        0.08234,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_rta_variable():
    ivc = get_indep_var_comp(list_inputs(ComputeRTAVariable()), __file__, XML_FILE)

    problem = run_system(ComputeRTAVariable(), ivc)

    assert problem.get_val("data:aerodynamics:low_speed:unit_reynolds") == pytest.approx(
        4107491.33,
        rel=1e-3,
    )
    assert problem.get_val("data:aerodynamics:cruise:unit_reynolds") == pytest.approx(
        5832253.82,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)
