import numpy as np
import pytest
import os.path as pth

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


def test_cl_ref():
    ivc = get_indep_var_comp(list_inputs(ClRef()), __file__, XML_FILE)

    ivc.add_output(
        "vector_product",
        np.linspace(2.6555, 1.6998, 100)
        * (np.ones(100) - np.linspace(0.0, 0.999993876, 100) ** 2.0),
    )

    problem = run_system(ClRef(), ivc)

    assert problem.get_val("data:aerodynamics:wing:low_speed:CL_ref") == pytest.approx(
        0.67888,
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
        0.67888,
        rel=1e-3,
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
    ) == pytest.approx(
        0.08234,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_simple_rta_variable():
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
