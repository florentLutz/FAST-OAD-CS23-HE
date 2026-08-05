# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
test_ducted_fan.py
====================
First unit tests for the ducted fan sizing and performance groups. Geometry and wing values in
data/reference_ducted_fan.xml are taken from a converged run of the AMPERE case
(integration_tests/ampere_final/, ducted_fan_1), so these tests exercise the model with a
physically plausible, real 40-fan DEP configuration.

Expected values below are the model's own current output (captured by running these groups with
the inputs described above), not externally validated reference data -- there is no independent
ground truth for these intermediate quantities. They serve as a regression baseline: if a future
change to the ducted fan model alters these values, that is a signal to check whether the change
was intentional, not necessarily an error.
"""

import os.path as pth

import numpy as np
import pytest

import openmdao.api as om

from ..sizing_ducted_fan_new import SizingDuctedFan
from ..perf_ducted_fan_new import PerformancesDuctedFan

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_ducted_fan.xml"
NB_POINTS_TEST = 2


def test_sizing_ducted_fan():
    ivc = get_indep_var_comp(
        list_inputs(SizingDuctedFan(ducted_fan_id="ducted_fan_1", position="on_the_wing")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDuctedFan(ducted_fan_id="ducted_fan_1", position="on_the_wing"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:mass", units="kg"
    ) == pytest.approx(1.828, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:CG:x", units="m"
    ) == pytest.approx(1.941, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:CG:y", units="m"
    ) == pytest.approx(2.232, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:wing_chord_ref", units="m"
    ) == pytest.approx(1.880, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:low_speed:CD0"
    ) == pytest.approx(4.025e-05, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:cruise:CD0"
    ) == pytest.approx(3.657e-05, rel=1e-2)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))

    problem.check_partials(compact_print=True)


def test_performance_ducted_fan():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesDuctedFan(ducted_fan_id="ducted_fan_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )

    # Two representative flight conditions taken from the AMPERE mission profile (low-speed
    # climb-out and cruise): true airspeed, altitude, density and alpha are the AMPERE case's own
    # values. Thrust is a representative assumed per-fan demand (not saved in the AMPERE mission
    # output, which only records rpm/shaft_power/torque downstream of it), chosen to fall well
    # inside the surrogate's training domain.
    ivc.add_output("true_airspeed", val=np.array([38.0, 77.16666666666667]), units="m/s")
    ivc.add_output("altitude", val=np.array([0.0, 2438.4]), units="m")
    ivc.add_output("density", val=np.array([1.2249908312142817, 0.962863065122232]), units="kg/m**3")
    ivc.add_output("alpha", val=np.array([7.002159739237071, 1.9672491429330536]), units="deg")
    ivc.add_output("thrust", val=np.array([100.0, 150.0]), units="N")

    problem = run_system(
        PerformancesDuctedFan(ducted_fan_id="ducted_fan_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("rpm", units="min**-1") == pytest.approx(
        [10401.7, 16787.4], rel=1e-2
    )
    assert problem.get_val("power_coefficient") == pytest.approx([1.0198, 1.1707], rel=1e-2)
    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        [6.356, 24.110], rel=1e-2
    )
    assert problem.get_val("torque_in", units="N*m") == pytest.approx([5.835, 13.715], rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:tip_mach_max"
    ) == pytest.approx(0.704, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:advance_ratio_max"
    ) == pytest.approx(1.103, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:rpm_max", units="min**-1"
    ) == pytest.approx(16787.4, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ducted_fan:ducted_fan_1:torque_max", units="N*m"
    ) == pytest.approx(13.715, rel=1e-2)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))

    problem.check_partials(compact_print=True)
