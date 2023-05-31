# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pytest

from ..components.cstr_enforce import ConstraintsInductorAirGapEnforce
from ..components.cstr_ensure import ConstraintsInductorAirGapEnsure

from ..components.sizing_inductor_energy import SizingInductorEnergy
from ..components.sizing_inductor_iron_surface import SizingInductorIronSurface
from ..components.sizing_inductor_reluctance import SizingInductorReluctance
from ..components.sizing_inductor_turn_number import SizingInductorTurnNumber
from ..components.sizing_inductor_copper_wire_area import SizingInductorCopperWireArea
from ..components.sizing_inductor_core_scaling import SizingInductorCoreScaling
from ..components.sizing_inductor_core_mass import SizingInductorCoreMass
from ..components.sizing_inductor_core_dimensions import SizingInductorCoreDimensions
from ..components.sizing_inductor_copper_mass import SizingInductorCopperMass
from ..components.sizing_inductor_mass import SizingInductorMass
from ..components.sizing_inductor_resistance import SizingInductorResistance
from ..components.sizing_inductor import SizingInductor

from fastga_he.powertrain_builder.powertrain import PT_DATA_PREFIX

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_inductor.xml"
PREFIX = PT_DATA_PREFIX + "DC_DC_converter:dc_dc_converter_1"


def test_constraints_air_gap_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsInductorAirGapEnforce(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsInductorAirGapEnforce(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:air_gap",
            units="m",
        )
        == pytest.approx(0.0126775, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_air_gap_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsInductorAirGapEnsure(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsInductorAirGapEnsure(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor"
            ":air_gap",
            units="m",
        )
        == pytest.approx(-0.0026775, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_mag_energy():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorEnergy(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorEnergy(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:magnetic_energy_rating",
            units="J",
        )
        == pytest.approx(35.76, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_iron_surface():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorIronSurface(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorIronSurface(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:iron_surface",
            units="mm**2",
        )
        == pytest.approx(15720.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_reluctance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorReluctance(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorReluctance(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:reluctance",
            units="H**-1",
        )
        / 1e6
        == pytest.approx(0.289, rel=1e-2)
    )

    # Partials are OK, it is just a question of step
    problem.check_partials(compact_print=True)


def test_inductor_turn_number():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorTurnNumber(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorTurnNumber(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:turn_number",
        )
        == pytest.approx(11.36, rel=1e-2)
    )

    # Partials are OK, it is just a question of step
    problem.check_partials(compact_print=True)


def test_inductor_copper_wire_area():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorCopperWireArea(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorCopperWireArea(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:wire_section_area",
        )
        == pytest.approx(8.0e-05, rel=1e-2)
    )

    # Partials are OK, it is just a question of step
    problem.check_partials(compact_print=True)


def test_inductor_core_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorCoreScaling(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorCoreScaling(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:scaling:core_mass",
        )
        == pytest.approx(98.30, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:scaling:core_dimension",
        )
        == pytest.approx(4.61, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_core_mass():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorCoreMass(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorCoreMass(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:core_mass",
            units="kg",
        )
        == pytest.approx(48.46, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_core_dimensions():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorCoreDimensions(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorCoreDimensions(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:core_dimension:B",
            units="m",
        )
        == pytest.approx(0.3372215, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:core_dimension:C",
            units="m",
        )
        == pytest.approx(0.126775, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_copper_mass():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorCopperMass(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorCopperMass(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:copper_mass",
            units="kg",
        )
        == pytest.approx(5.16, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_mass():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorMass(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorMass(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:mass",
            units="kg",
        )
        == pytest.approx(201.76, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductorResistance(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductorResistance(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:resistance",
            units="ohm",
        )
        == pytest.approx(0.00183188, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_sizing():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductor(prefix=PREFIX)),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInductor(prefix=PREFIX), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:resistance",
            units="ohm",
        )
        == pytest.approx(0.00183188, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:mass",
            units="kg",
        )
        == pytest.approx(102.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)
