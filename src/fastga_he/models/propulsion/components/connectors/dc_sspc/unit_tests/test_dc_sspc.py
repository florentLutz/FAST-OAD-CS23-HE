# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_resistance_scaling import SizingDCSSPCResistanceScaling
from ..components.sizing_reference_resistance import SizingDCSSPCResistances
from ..components.sizing_weight import SizingDCSSPCWeight
from ..components.perf_resistance import PerformancesDCSSPCResistance

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_dc_sspc.xml"
NB_POINTS_TEST = 10

# Note: The PerformancesConverterRelations, PerformancesConverterGeneratorSide and
# PerformancesDCDCConverter components cannot be easily tested on its own as it is meant to act
# jointly with the other components of the converter, so it won't be tested here but rather in
# the assemblies


def test_energy_coefficients_scaling_ratio():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSSPCResistanceScaling(dc_sspc_id="dc_sspc_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingDCSSPCResistanceScaling(dc_sspc_id="dc_sspc_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:scaling:resistance"
    ) == pytest.approx(1.125, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSSPCResistances(dc_sspc_id="dc_sspc_1")), __file__, XML_FILE
    )

    problem = run_system(SizingDCSSPCResistances(dc_sspc_id="dc_sspc_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:igbt:resistance", units="ohm"
    ) == pytest.approx(0.00169875, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:diode:resistance", units="ohm"
    ) == pytest.approx(0.00210375, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSSPCWeight(dc_sspc_id="dc_sspc_1")), __file__, XML_FILE
    )

    problem = run_system(SizingDCSSPCWeight(dc_sspc_id="dc_sspc_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:mass", units="kg"
    ) == pytest.approx(10.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_resistance():

    # Research independent input value in .xml file, should be the same regardless of option
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesDCSSPCResistance(
                dc_sspc_id="dc_sspc_1", number_of_points=NB_POINTS_TEST, closed=True
            )
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PerformancesDCSSPCResistance(
            dc_sspc_id="dc_sspc_1", number_of_points=NB_POINTS_TEST, closed=True
        ),
        ivc,
    )
    expected_resistance = np.full(NB_POINTS_TEST, 0.00380)
    assert problem.get_val("resistance_sspc", units="ohm") == pytest.approx(
        expected_resistance, rel=1e-2
    )

    problem = run_system(
        PerformancesDCSSPCResistance(
            dc_sspc_id="dc_sspc_1", number_of_points=NB_POINTS_TEST, closed=False
        ),
        ivc,
    )
    expected_resistance = np.full(NB_POINTS_TEST, np.inf)
    assert problem.get_val("resistance_sspc", units="ohm") == pytest.approx(
        expected_resistance, rel=1e-2
    )
