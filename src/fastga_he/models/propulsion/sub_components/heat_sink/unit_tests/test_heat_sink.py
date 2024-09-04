# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import openmdao.api as om
import pytest

from ..components.sizing_heat_sink_dimension import SizingHeatSinkDimension
from ..components.sizing_heat_sink_tube_length import SizingHeatSinkTubeLength
from ..components.sizing_heat_sink_tube_mass_flow import SizingHeatSinkTubeMassFlow
from ..components.sizing_heat_sink_coolant_prandtl import SizingHeatSinkCoolantPrandtl
from ..components.sizing_heat_sink_tube_inner_diameter import (
    SizingHeatSinkTubeInnerDiameter,
)
from ..components.sizing_heat_sink_tube_outer_diameter import (
    SizingHeatSinkTubeOuterDiameter,
)
from ..components.sizing_heat_sink_tube_weight import SizingHeatSinkTubeWeight
from ..components.sizing_heat_sink_height import SizingHeatSinkHeight
from ..components.sizing_heat_sink_weight import SizingHeatSinkWeight
from ..components.sizing_heat_sink import SizingHeatSink

from fastga_he.powertrain_builder.powertrain import PT_DATA_PREFIX

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_heat_sink.xml"
PREFIX = PT_DATA_PREFIX + "inverter:inverter_1"


def test_dimension_heat_sink():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSinkDimension(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingHeatSinkDimension(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":heat_sink:length", units="m") == pytest.approx(
        0.2145, rel=1e-2
    )
    assert problem.get_val(PREFIX + ":heat_sink:width", units="m") == pytest.approx(
        0.1716, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_length_heat_sink_tube():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSinkTubeLength(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingHeatSinkTubeLength(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":heat_sink:tube:length", units="m") == pytest.approx(
        0.858, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_mass_flow_heat_sink_tube():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSinkTubeMassFlow(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingHeatSinkTubeMassFlow(prefix=PREFIX), ivc)

    assert problem.get_val(
        PREFIX + ":heat_sink:coolant:max_mass_flow",
        units="m**3/s",
    ) == pytest.approx(11.1e-5, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_mass_flow_heat_sink_coolant_prandtl():
    problem = om.Problem(reports=False)
    model = problem.model
    model.add_subsystem("component", SizingHeatSinkCoolantPrandtl(prefix=PREFIX), promotes=["*"])
    problem.setup()
    problem.run_model()

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:coolant:Prandtl_number",
    ) == pytest.approx(39.5, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_tube_inner_diameter():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSinkTubeInnerDiameter(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingHeatSinkTubeInnerDiameter(prefix=PREFIX), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:tube:inner_diameter",
        units="mm",
    ) == pytest.approx(1.27, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_tube_outer_diameter():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSinkTubeOuterDiameter(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingHeatSinkTubeOuterDiameter(prefix=PREFIX), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:tube:outer_diameter",
        units="mm",
    ) == pytest.approx(3.77, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_tube_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSinkTubeWeight(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingHeatSinkTubeWeight(prefix=PREFIX), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:tube:mass",
        units="kg",
    ) == pytest.approx(0.083, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_height():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSinkHeight(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingHeatSinkHeight(prefix=PREFIX), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:height",
        units="mm",
    ) == pytest.approx(5.655, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSinkWeight(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingHeatSinkWeight(prefix=PREFIX), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:mass",
        units="kg",
    ) == pytest.approx(0.619, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSink(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingHeatSink(prefix=PREFIX), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:mass",
        units="kg",
    ) == pytest.approx(0.634, rel=1e-2)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))
