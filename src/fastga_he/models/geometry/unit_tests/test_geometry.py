# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import pytest

from ..aspect_ratio_fixed_span import AspectRatioFromTargetSpan
from ..tail_sizing import TailAreasFromVolume

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")

XML_FILE = "data.xml"


def test_aspect_ratio_from_fixed_span():
    ivc = get_indep_var_comp(list_inputs(AspectRatioFromTargetSpan()), __file__, XML_FILE)

    problem = run_system(AspectRatioFromTargetSpan(), ivc)

    assert problem.get_val("data:geometry:wing:aspect_ratio") == pytest.approx(16.0, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_volumetric_coefficient_tail_sizing():
    ivc = get_indep_var_comp(list_inputs(TailAreasFromVolume()), __file__, XML_FILE)

    problem = run_system(TailAreasFromVolume(), ivc)

    assert problem.get_val("data:geometry:horizontal_tail:area", units="m**2") == pytest.approx(
        2.48, rel=1e-3
    )
    assert problem.get_val("data:geometry:vertical_tail:area", units="m**2") == pytest.approx(
        2.87, rel=1e-3
    )

    problem.check_partials(compact_print=True)
