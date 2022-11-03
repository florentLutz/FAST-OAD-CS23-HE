# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.perf_load_side import PerformancesConverterLoadSide
from ..components.perf_dc_dc_converter import PerformancesDCDCConverter

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_dc_bus.xml"
NB_POINTS_TEST = 10

# Note: The PerformancesConverterRelations, PerformancesConverterGeneratorSide and
# PerformancesDCDCConverter components cannot be easily tested on its own as it is meant to act
# jointly with the other components of the converter, so it won't be tested here but rather in
# the assemblies


def test_load_side():

    ivc = om.IndepVarComp()
    power = np.linspace(350, 400, NB_POINTS_TEST)
    ivc.add_output("power", power, units="kW")
    voltage_in = np.full(NB_POINTS_TEST, 810)
    ivc.add_output("voltage_in", voltage_in, units="V")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesConverterLoadSide(number_of_points=NB_POINTS_TEST), ivc)

    expected_current = np.array(
        [432.1, 439.0, 445.8, 452.7, 459.5, 466.4, 473.3, 480.1, 487.0, 493.8]
    )
    assert problem.get_val("current_in", units="A") == pytest.approx(expected_current, rel=1e-2)

    problem.check_partials(compact_print=True)
