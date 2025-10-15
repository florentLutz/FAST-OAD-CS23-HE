# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth

import pytest

from ..components.cd0_power_train import Cd0PowerTrain
from ..components.cd0_wing_rta import (
    _SweepCorrection,
    _CamberContribution,
    _RelativeThicknessContribution,
    _Cd0Wing,
    Cd0Wing,
)
from ..components.cd0_total_rta import _TotalCd0ParasiticFactor, _AircraftCd0, Cd0Total
from fastga_he.models.aerodynamics.components.flat_plate_friction_drag_coeff import (
    FlatPlateFrictionDragCoefficient,
)

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")

XML_FILE = "data.xml"


def test_powertrain_cd0():
    ivc = get_indep_var_comp(list_inputs(Cd0PowerTrain()), __file__, XML_FILE)

    problem = run_system(Cd0PowerTrain(), ivc)

    assert problem.get_val("data:aerodynamics:nacelles:cruise:CD0") == pytest.approx(
        0.0008112,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_flat_plate_friction():
    ivc = get_indep_var_comp(list_inputs(FlatPlateFrictionDragCoefficient()), __file__, XML_FILE)

    problem = run_system(FlatPlateFrictionDragCoefficient(), ivc)

    assert problem.get_val("plate_drag_friction_coeff") == pytest.approx(
        0.002794,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_thickness_contribution():
    ivc = get_indep_var_comp(list_inputs(_RelativeThicknessContribution()), __file__, XML_FILE)

    problem = run_system(_RelativeThicknessContribution(), ivc)

    assert problem.get_val("thickness_contribution") == pytest.approx(
        0.53528,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_camber_contribution():
    ivc = get_indep_var_comp(list_inputs(_CamberContribution()), __file__, XML_FILE)

    problem = run_system(_CamberContribution(), ivc)

    assert problem.get_val("camber_contribution") == pytest.approx(
        0.5035,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_sweep_contribution():
    ivc = get_indep_var_comp(list_inputs(_SweepCorrection()), __file__, XML_FILE)

    problem = run_system(_SweepCorrection(), ivc)

    assert problem.get_val("sweep_correction") == pytest.approx(
        0.9997,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_cd0_wing():
    ivc = get_indep_var_comp(list_inputs(_Cd0Wing()), __file__, XML_FILE)
    ivc.add_output("plate_drag_friction_coeff", val=0.002794)
    ivc.add_output("thickness_contribution", val=0.53528)
    ivc.add_output("camber_contribution", val=0.5035)
    ivc.add_output("sweep_correction", val=0.9997)

    problem = run_system(_Cd0Wing(), ivc)

    assert problem.get_val("data:aerodynamics:wing:cruise:CD0") == pytest.approx(
        0.01053,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_cd0_wing_overall():
    ivc = get_indep_var_comp(list_inputs(Cd0Wing()), __file__, XML_FILE)

    problem = run_system(Cd0Wing(), ivc)

    assert problem.get_val("data:aerodynamics:wing:cruise:CD0") == pytest.approx(
        0.01053,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_parasitic_factor():
    ivc = get_indep_var_comp(list_inputs(_TotalCd0ParasiticFactor()), __file__, XML_FILE)

    problem = run_system(_TotalCd0ParasiticFactor(), ivc)

    assert problem.get_val("k_parasitic") == pytest.approx(
        0.1321,
        rel=1e-3,
    )
    assert problem.get_val("CD0_fuselage") == pytest.approx(
        0.00863,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_cd0_aircraft():
    ivc = get_indep_var_comp(list_inputs(_AircraftCd0()), __file__, XML_FILE)
    ivc.add_output("k_parasitic", val=0.1321)
    ivc.add_output("CD0_fuselage", val=0.00863)

    problem = run_system(_AircraftCd0(), ivc)

    assert problem.get_val("data:aerodynamics:aircraft:cruise:CD0:parasitic") == pytest.approx(
        0.00311,
        rel=1e-3,
    )
    assert problem.get_val("data:aerodynamics:aircraft:cruise:CD0") == pytest.approx(
        0.02664,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_cd0_aircraft_overall():
    ivc = get_indep_var_comp(list_inputs(Cd0Total()), __file__, XML_FILE)

    problem = run_system(Cd0Total(), ivc)

    assert problem.get_val("data:aerodynamics:aircraft:cruise:CD0") == pytest.approx(
        0.02664,
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)
