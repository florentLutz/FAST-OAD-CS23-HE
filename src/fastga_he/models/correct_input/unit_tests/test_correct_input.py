import numpy as np
import pytest
import os.path as pth
import openmdao.api as om

from ..aero_approximated import (
    _LengthVector,
    _Ly,
    _VectorProduct,
    _Cl_Ref,
    _Induced_Drag_Coefficient,
    Aero_compute,
)

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system
from fastoad_cs25.models.aerodynamics.constants import PolarType

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")

XML_FILE = "data.xml"


def test_length_vector():
    ivc = get_indep_var_comp(list_inputs(_LengthVector()), __file__, XML_FILE)

    problem = run_system(_LengthVector(), ivc)

    assert problem.get_val("half_wing_coordinate") == pytest.approx(
        np.linspace(0.0, 13.5335, 100),
        rel=1e-3,
    )
    assert problem.get_val("chord_vector") == pytest.approx(
        np.linspace(2.6555, 1.6998, 100),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_l_y():
    ivc = get_indep_var_comp(list_inputs(_Ly()), __file__, XML_FILE)

    ivc.add_output("half_wing_coordinate", np.linspace(0.0, 13.5335, 100))

    problem = run_system(_Ly(), ivc)

    assert problem.get_val("l_y") == pytest.approx(
        np.ones(100) - np.linspace(0.0, 0.999993876, 100) ** 2.0,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_vector_product():
    ivc = om.IndepVarComp()

    ivc.add_output("chord_vector", np.linspace(2.6555, 1.6998, 100))
    ivc.add_output(
        "l_y",
        np.ones(100) - np.linspace(0.0, 0.999993876, 100) ** 2.0,
    )

    problem = run_system(_VectorProduct(), ivc)

    assert problem.get_val("vector_product") == pytest.approx(
        np.linspace(2.6555, 1.6998, 100)
        * (np.ones(100) - np.linspace(0.0, 0.999993876, 100) ** 2.0),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_cl_ref():
    ivc = get_indep_var_comp(list_inputs(_Cl_Ref()), __file__, XML_FILE)

    ivc.add_output(
        "vector_product",
        np.linspace(2.6555, 1.6998, 100)
        * (np.ones(100) - np.linspace(0.0, 0.999993876, 100) ** 2.0),
    )

    problem = run_system(_Cl_Ref(), ivc)

    assert problem.get_val("data:aerodynamics:wing:low_speed:CL_ref") == pytest.approx(
        0.67888,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_induced_drag():
    ivc = get_indep_var_comp(list_inputs(_Induced_Drag_Coefficient()), __file__, XML_FILE)

    problem = run_system(_Induced_Drag_Coefficient(), ivc)

    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
    ) == pytest.approx(
        0.08234,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)

def test_aero_approx():
    ivc = get_indep_var_comp(list_inputs(Aero_compute()), __file__, XML_FILE)

    problem = run_system(Aero_compute(), ivc)

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
