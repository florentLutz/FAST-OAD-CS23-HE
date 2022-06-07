"""
Test module for sample discipline
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pytest

from ..sample_discipline import SampleDiscipline

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "data.xml"


def test_sample_discipline():
    """Tests computation of the sample discipline."""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(SampleDiscipline()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SampleDiscipline(), ivc)
    sample_output = problem.get_val("sample_output", units="kg")
    assert sample_output == pytest.approx(4.0, abs=1e-3)
