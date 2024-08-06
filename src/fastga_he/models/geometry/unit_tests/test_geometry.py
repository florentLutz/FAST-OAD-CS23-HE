# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import pytest

from ..aspect_ratio_fixed_span import AspectRatioFromTargetSpan

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")

XML_FILE = "data.xml"


def test_aspect_ratio_from_fixed_span():
    ivc = get_indep_var_comp(list_inputs(AspectRatioFromTargetSpan()), __file__, XML_FILE)

    problem = run_system(AspectRatioFromTargetSpan(), ivc)

    assert problem.get_val("data:geometry:wing:aspect_ratio") == pytest.approx(10.39, rel=1e-3)

    problem.check_partials(compact_print=True)
