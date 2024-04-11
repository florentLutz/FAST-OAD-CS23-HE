# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pytest

from ..components.sizing_turboshaft_uninstalled_weight import SizingTurboshaftUninstalledWeight
from ..components.sizing_turboshaft_weight import SizingTurboshaftWeight
from ..components.sizing_turboshaft_dimensions import SizingTurboshaftDimensions

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_turboshaft.xml"
NB_POINTS_TEST = 10


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
