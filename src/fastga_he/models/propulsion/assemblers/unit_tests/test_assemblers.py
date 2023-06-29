# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..delta_from_pt_file import SlipstreamAirframeLiftClean, SlipstreamAirframeLift

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_data.xml"
NB_POINTS_TEST = 10


def test_clean_lift_wing():

    ivc = get_indep_var_comp(
        list_inputs(SlipstreamAirframeLiftClean(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("alpha", val=np.full(NB_POINTS_TEST, 5.0), units="deg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamAirframeLiftClean(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("cl_wing_clean") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.6533), rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_airframe_lift():

    flaps_positions = ["cruise", "landing", "takeoff"]
    expected_values = (
        np.full(NB_POINTS_TEST, 0.6533),
        np.full(NB_POINTS_TEST, 1.2620),
        np.full(NB_POINTS_TEST, 0.9536),
    )

    for flaps_position, expected_value in zip(flaps_positions, expected_values):

        ivc = get_indep_var_comp(
            list_inputs(
                SlipstreamAirframeLift(
                    number_of_points=NB_POINTS_TEST,
                    flaps_position=flaps_position,
                )
            ),
            __file__,
            XML_FILE,
        )
        ivc.add_output("cl_wing_clean", val=np.full(NB_POINTS_TEST, 0.6533), units="deg")

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SlipstreamAirframeLift(
                number_of_points=NB_POINTS_TEST,
                flaps_position=flaps_position,
            ),
            ivc,
        )

        assert problem.get_val("cl_airframe") == pytest.approx(expected_value, rel=1e-3)

        problem.check_partials(compact_print=True)
