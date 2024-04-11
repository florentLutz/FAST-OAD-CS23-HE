# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pytest

from ..components.cstr_enforce import ConstraintsRatedPowerEnforce
from ..components.cstr_ensure import ConstraintsRatedPowerEnsure
from ..components.cstr_turboshaft import ConstraintTurboshaftPowerRateMission

from ..components.sizing_turboshaft_uninstalled_weight import SizingTurboshaftUninstalledWeight
from ..components.sizing_turboshaft_weight import SizingTurboshaftWeight
from ..components.sizing_turboshaft_dimensions import SizingTurboshaftDimensions
from ..components.sizing_turboshaft_nacelle_dimensions import SizingTurboshaftNacelleDimensions
from ..components.sizing_turboshaft_nacelle_wet_area import SizingTurboshaftNacelleWetArea
from ..components.sizing_turboshaft_drag import SizingTurboshaftDrag
from ..components.sizing_turboshaft_cg_x import SizingTurboshaftCGX
from ..components.sizing_turboshaft_cg_y import SizingTurboshaftCGY

from ..components.sizing_turboshaft import SizingTurboshaft

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_turboshaft.xml"
NB_POINTS_TEST = 10


def test_constraint_power_enforce():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRatedPowerEnforce(turboshaft_id="turboshaft_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRatedPowerEnforce(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating", units="kW"
    ) == pytest.approx(625, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_ensure():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRatedPowerEnsure(turboshaft_id="turboshaft_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRatedPowerEnsure(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating", units="kW"
    ) == pytest.approx(-9.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintTurboshaftPowerRateMission(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintTurboshaftPowerRateMission(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:shaft_power_rating", units="kW"
    ) == pytest.approx(634.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_uninstalled_weight():

    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftUninstalledWeight(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftUninstalledWeight(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:uninstalled_mass", units="kg"
    ) == pytest.approx(192.9758, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_installed_weight():

    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftWeight(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftWeight(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:mass", units="kg"
    ) == pytest.approx(231.57096, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_turboshaft_dimensions():

    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftDimensions(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftDimensions(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:engine:height", units="m"
    ) == pytest.approx(0.512, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:engine:width", units="m"
    ) == pytest.approx(0.512, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:engine:length", units="m"
    ) == pytest.approx(1.702, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_turboshaft_nacelle_dimensions():

    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftNacelleDimensions(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftNacelleDimensions(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:height", units="m"
    ) == pytest.approx(0.5632, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:width", units="m"
    ) == pytest.approx(0.5632, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:length", units="m"
    ) == pytest.approx(3.404, rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc_bis = get_indep_var_comp(
        list_inputs(
            SizingTurboshaftNacelleDimensions(turboshaft_id="turboshaft_1", position="in_the_front")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingTurboshaftNacelleDimensions(turboshaft_id="turboshaft_1", position="in_the_front"),
        ivc_bis,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:height", units="m"
    ) == pytest.approx(0.5632, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:width", units="m"
    ) == pytest.approx(0.5632, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:length", units="m"
    ) == pytest.approx(1.9573, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_nacelle_wet_area():

    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaftNacelleWetArea(turboshaft_id="turboshaft_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaftNacelleWetArea(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:nacelle:wet_area", units="m**2"
    ) == pytest.approx(7.6685312, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_nacelle_drag():

    expected_drag_ls = [4.26, 0.0, 0.0]
    expected_drag_cruise = [4.217, 0.0, 0.0]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_drag_ls, expected_drag_cruise
    ):
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingTurboshaftDrag(
                        turboshaft_id="turboshaft_1", position=option, low_speed_aero=ls_option
                    )
                ),
                __file__,
                XML_FILE,
            )
            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingTurboshaftDrag(
                    turboshaft_id="turboshaft_1", position=option, low_speed_aero=ls_option
                ),
                ivc,
            )

            if ls_option:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:turboshaft:turboshaft_1:low_speed:CD0",
                    )
                    * 1e3
                    == pytest.approx(ls_drag, rel=1e-2)
                )
            else:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:turboshaft:turboshaft_1:cruise:CD0",
                    )
                    * 1e3
                    == pytest.approx(cruise_drag, rel=1e-2)
                )

            # Slight error on reynolds is due to step
            problem.check_partials(compact_print=True)


def test_turboshaft_cg_x():

    expected_cg = [3.703, 1.702, 3.679]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):

        ivc = get_indep_var_comp(
            list_inputs(SizingTurboshaftCGX(turboshaft_id="turboshaft_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingTurboshaftCGX(turboshaft_id="turboshaft_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_turboshaft_cg_y():

    expected_cg = [1.7, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):

        ivc = get_indep_var_comp(
            list_inputs(SizingTurboshaftCGY(turboshaft_id="turboshaft_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingTurboshaftCGY(turboshaft_id="turboshaft_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_ice_sizing():

    ivc = get_indep_var_comp(
        list_inputs(SizingTurboshaft(turboshaft_id="turboshaft_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingTurboshaft(turboshaft_id="turboshaft_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:mass", units="kg"
    ) == pytest.approx(231.57096, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:CG:x", units="m"
    ) == pytest.approx(3.703, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:CG:y", units="m"
    ) == pytest.approx(1.7, rel=1e-2)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:low_speed:CD0",
        )
        * 1e3
        == pytest.approx(4.273, rel=1e-2)
    )

    problem.check_partials(compact_print=True)
