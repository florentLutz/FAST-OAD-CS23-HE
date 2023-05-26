# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pytest
import openmdao.api as om

from ..components.sizing_capacitor_capacity_scaling import SizingCapacitorCapacityScaling
from ..components.sizing_capacitor_diameter_scaling import SizingCapacitorDiameterScaling
from ..components.sizing_capacitor_diameter import SizingCapacitorDiameter
from ..components.sizing_capacitor_height import SizingCapacitorHeight
from ..components.sizing_capacitor_height_scaling import SizingCapacitorHeightScaling
from ..components.sizing_capacitor_thermal_resistance_scaling import (
    SizingCapacitorThermalResistanceScaling,
)
from ..components.sizing_capacitor_thermal_resistance import SizingCapacitorThermalResistance
from ..components.sizing_capacitor_resistance_scaling import SizingCapacitorResistanceScaling
from ..components.sizing_capacitor_resistance import SizingCapacitorResistance
from ..components.sizing_capacitor_mass_scaling import SizingCapacitorMassScaling
from ..components.sizing_capacitor_mass import SizingCapacitorMass

from ..components.sizing_capacitor import SizingCapacitor

from fastga_he.powertrain_builder.powertrain import PT_DATA_PREFIX

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_capacitor.xml"
PREFIX = PT_DATA_PREFIX + "inverter:inverter_1"


def test_capacity_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorCapacityScaling(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorCapacityScaling(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:scaling:capacity") == pytest.approx(2.02, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diameter_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorDiameterScaling(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorDiameterScaling(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:scaling:diameter") == pytest.approx(1.59, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diameter():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorDiameter(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorDiameter(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:diameter", units="m") == pytest.approx(
        0.159, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_height():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorHeight(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorHeight(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:height", units="m") == pytest.approx(
        0.318, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_height_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorHeightScaling(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorHeightScaling(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:scaling:height") == pytest.approx(2.052, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_thermal_resistance_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorThermalResistanceScaling(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorThermalResistanceScaling(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:scaling:thermal_resistance") == pytest.approx(
        0.58, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_thermal_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorThermalResistance(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorThermalResistance(prefix=PREFIX), ivc)

    assert problem.get_val(
        PREFIX + ":capacitor:thermal_resistance", units="degK/W"
    ) == pytest.approx(1.74, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorResistanceScaling(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorResistanceScaling(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:scaling:resistance") == pytest.approx(
        0.396, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorResistance(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorResistance(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:resistance", units="mohm") == pytest.approx(
        1.28, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_mass_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorMassScaling(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorMassScaling(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:scaling:mass") == pytest.approx(5.18, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_mass():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitorMass(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingCapacitorMass(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:mass", units="kg") == pytest.approx(7.77, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_full_sizing_capacitor():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(PREFIX + ":capacitor:capacity", units="F", val=2.27e-04)
    ivc.add_output(PREFIX + ":capacitor:aspect_ratio", val=1.0)

    problem = run_system(SizingCapacitor(prefix=PREFIX), ivc)

    assert problem.get_val(PREFIX + ":capacitor:mass", units="kg") == pytest.approx(0.22, rel=1e-2)
    assert problem.get_val(PREFIX + ":capacitor:resistance", units="mohm") == pytest.approx(
        8.6, rel=1e-2
    )
    assert problem.get_val(
        PREFIX + ":capacitor:thermal_resistance", units="degK/W"
    ) == pytest.approx(5.69, rel=1e-2)

    problem.check_partials(compact_print=True)
