# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import pytest

import fastoad.api as oad

from ..components.sizing_radius import SizingPropellerRadius
from ..components.sizing_propeller_section_aero import SizingPropellerSectionAero
from ..components.sizing_sweep import SizingPropellerSweep
from ..components.sizing_chord import SizingPropellerChord
from ..components.sizing_twist import SizingPropellerTwist
from ..components.perf_propeller import PerformancesPropeller

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_propeller.xml"

NB_POINTS_TEST = 1


def test_compute_elements_radius():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPropellerRadius(propeller_id="propeller_1")), __file__, XML_FILE
    )

    problem = run_system(SizingPropellerRadius(propeller_id="propeller_1"), ivc)

    expected_radius = np.array([0.255, 0.364, 0.473, 0.582, 0.692, 0.801, 0.91])

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:elements_radius", units="m"
    ) == pytest.approx(expected_radius, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_compute_elements_aero_profile():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPropellerSectionAero(propeller_id="propeller_1")), __file__, XML_FILE
    )

    _ = run_system(SizingPropellerSectionAero(propeller_id="propeller_1"), ivc)


def test_sweep():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPropellerSweep(propeller_id="propeller_1")), __file__, XML_FILE
    )

    problem = run_system(SizingPropellerSweep(propeller_id="propeller_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:elements_sweep", units="deg"
    ) == pytest.approx(np.zeros(7), rel=1e-2)


def test_chord():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPropellerChord(propeller_id="propeller_1")), __file__, XML_FILE
    )

    problem = run_system(SizingPropellerChord(propeller_id="propeller_1"), ivc)

    expected_chord = np.array([0.146, 0.161, 0.172, 0.199, 0.203, 0.186, 0.129])
    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:elements_chord", units="m"
    ) == pytest.approx(expected_chord, rel=1e-2)


def test_twist():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPropellerTwist(propeller_id="propeller_1")), __file__, XML_FILE
    )

    problem = run_system(SizingPropellerTwist(propeller_id="propeller_1"), ivc)

    expected_twist = np.array([56.0, 52.5, 49.6, 47.4, 45.5, 44.0, 42.8])
    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:elements_twist", units="deg"
    ) == pytest.approx(expected_twist, rel=1e-2)


def test_propeller_performances():

    problem = oad.FASTOADProblem()
    model = problem.model
    model.add_subsystem(
        "radius",
        SizingPropellerRadius(propeller_id="propeller_1", elements_number=3),
        promotes=["*"],
    )
    model.add_subsystem(
        "chord",
        SizingPropellerChord(propeller_id="propeller_1", elements_number=3),
        promotes=["*"],
    )
    model.add_subsystem(
        "twist",
        SizingPropellerTwist(propeller_id="propeller_1", elements_number=3),
        promotes=["*"],
    )
    model.add_subsystem(
        "sweep",
        SizingPropellerSweep(propeller_id="propeller_1", elements_number=3),
        promotes=["*"],
    )
    model.add_subsystem(
        "aero_perf",
        SizingPropellerSectionAero(
            propeller_id="propeller_1",
            elements_number=3,
            sections_profile_name_list=["naca4430"],
            sections_profile_position_list=[0],
        ),
        promotes=["*"],
    )
    model.add_subsystem(
        "propeller_perf",
        PerformancesPropeller(
            propeller_id="propeller_1", number_of_points=NB_POINTS_TEST, elements_number=3
        ),
        promotes=["*"],
    )

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(model),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "propeller_rpm", val=np.full(NB_POINTS_TEST, 2500 * 2.0 * np.pi / 60.0), units="rad/s"
    )
    ivc.add_output("true_airspeed", val=np.full(NB_POINTS_TEST, 25.8385), units="m/s")
    ivc.add_output("twist_75", val=np.full(NB_POINTS_TEST, -35.50148156495961), units="deg")
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")

    problem_2 = oad.FASTOADProblem()
    model_2 = problem_2.model

    model_2.add_subsystem("ivc", ivc, promotes=["*"])
    model_2.add_subsystem("actual_model", model, promotes=["*"])

    problem_2.setup()
    # om.n2(problem_2)
    problem_2.run_model()

    print(problem_2["v_i"])
