# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_energy_coefficient_scaling import (
    SizingDCDCConverterEnergyCoefficientScaling,
)
from ..components.sizing_energy_coefficients import SizingDCDCConverterEnergyCoefficients
from ..components.sizing_resistance_scaling import SizingDCDCConverterResistanceScaling
from ..components.sizing_reference_resistance import SizingDCDCConverterResistances
from ..components.sizing_inductor_inductance import SizingDCDCConverterInductorInductance
from ..components.sizing_inductor_energy import SizingDCDCConverterInductorEnergy
from ..components.sizing_inductor_iron_surface import SizingDCDCConverterInductorIronSurface
from ..components.sizing_inductor_reluctance import SizingDCDCConverterInductorReluctance
from ..components.sizing_inductor_turn_number import SizingDCDCConverterInductorTurnNumber
from ..components.sizing_inductor_copper_wire_area import SizingDCDCConverterInductorCopperWireArea
from ..components.sizing_inductor_core_scaling import SizingDCDCConverterInductorCoreScaling
from ..components.sizing_inductor_core_mass import SizingDCDCConverterInductorCoreMass
from ..components.sizing_inductor_core_dimensions import SizingDCDCConverterInductorCoreDimensions
from ..components.sizing_inductor_copper_mass import SizingDCDCConverterInductorCopperMass
from ..components.sizing_inductor_mass import SizingDCDCConverterInductorMass
from ..components.sizing_inductor_resistance import SizingDCDCConverterInductorResistance
from ..components.sizing_inductor import SizingDCDCConverterInductor
from ..components.sizing_capacitor_capacity import SizingDCDCConverterCapacitorCapacity
from ..components.sizing_capacitor_weight import SizingDCDCConverterCapacitorWeight
from ..components.sizing_module_mass import SizingDCDCConverterCasingWeight
from ..components.sizing_weight import SizingDCDCConverterWeight, SizingDCDCConverterWeightBySum
from ..components.sizing_dc_dc_converter_cg import SizingDCDCConverterCG
from ..components.perf_switching_frequency import PerformancesSwitchingFrequencyMission
from ..components.perf_voltage_out_target import PerformancesVoltageOutTargetMission
from ..components.perf_load_side import PerformancesConverterLoadSide
from ..components.perf_duty_cycle import PerformancesDutyCycle
from ..components.perf_currents import PerformancesCurrents
from ..components.perf_conduction_losses import PerformancesConductionLosses
from ..components.perf_switching_losses import PerformancesSwitchingLosses
from ..components.perf_total_losses import PerformancesLosses
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_maximum import PerformancesMaximum

from ..components.cstr_enforce import (
    ConstraintsCurrentCapacitorEnforce,
    ConstraintsCurrentInductorEnforce,
    ConstraintsCurrentModuleEnforce,
    ConstraintsCurrentInputEnforce,
    ConstraintsVoltageEnforce,
    ConstraintsVoltageInputEnforce,
    ConstraintsFrequencyEnforce,
    ConstraintsInductorAirGapEnforce,
)
from ..components.cstr_ensure import (
    ConstraintsCurrentCapacitorEnsure,
    ConstraintsCurrentInductorEnsure,
    ConstraintsCurrentModuleEnsure,
    ConstraintsCurrentInputEnsure,
    ConstraintsVoltageEnsure,
    ConstraintsVoltageInputEnsure,
    ConstraintsFrequencyEnsure,
    ConstraintsInductorAirGapEnsure,
)

from ..components.sizing_dc_dc_converter import SizingDCDCConverter

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_dc_dc_converter.xml"
NB_POINTS_TEST = 10

# Note: The PerformancesConverterRelations, PerformancesConverterGeneratorSide and
# PerformancesDCDCConverter components cannot be easily tested on its own as it is meant to act
# jointly with the other components of the converter, so it won't be tested here but rather in
# the assemblies


def test_energy_coefficients_scaling_ratio():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingDCDCConverterEnergyCoefficientScaling(dc_dc_converter_id="dc_dc_converter_1")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingDCDCConverterEnergyCoefficientScaling(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:scaling:a"
    ) == pytest.approx(1.125, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:scaling:c"
    ) == pytest.approx(0.888, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_energy_coefficient():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterEnergyCoefficients(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterEnergyCoefficients(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_on:a"
    ) == pytest.approx(0.02379429, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_rr:a"
    ) == pytest.approx(0.00637224, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_off:a"
    ) == pytest.approx(0.02261044, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_on:b"
    ) == pytest.approx(3.326e-05, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_rr:b"
    ) == pytest.approx(0.000340, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_off:b"
    ) == pytest.approx(0.000254, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_on:c"
    ) == pytest.approx(3.420e-7, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_rr:c"
    ) == pytest.approx(-3.005e-8, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_off:c"
    ) == pytest.approx(-1.158e-7, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterResistanceScaling(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterResistanceScaling(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:scaling:resistance"
    ) == pytest.approx(1.125, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterResistances(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterResistances(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:igbt:resistance",
            units="ohm",
        )
        == pytest.approx(0.002265, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:diode:resistance",
            units="ohm",
        )
        == pytest.approx(0.002805, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_inductance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterInductorInductance(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorInductance(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:inductance",
            units="mH",
        )
        == pytest.approx(0.447, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_mag_energy():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterInductorEnergy(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorEnergy(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(SizingDCDCConverterInductorIronSurface(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorIronSurface(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(SizingDCDCConverterInductorReluctance(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorReluctance(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(SizingDCDCConverterInductorTurnNumber(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorTurnNumber(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(
            SizingDCDCConverterInductorCopperWireArea(dc_dc_converter_id="dc_dc_converter_1")
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorCopperWireArea(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(SizingDCDCConverterInductorCoreScaling(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorCoreScaling(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(SizingDCDCConverterInductorCoreMass(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorCoreMass(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(
            SizingDCDCConverterInductorCoreDimensions(dc_dc_converter_id="dc_dc_converter_1")
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorCoreDimensions(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(SizingDCDCConverterInductorCopperMass(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorCopperMass(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(SizingDCDCConverterInductorMass(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorMass(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(SizingDCDCConverterInductorResistance(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorResistance(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

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
        list_inputs(SizingDCDCConverterInductor(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingDCDCConverterInductor(dc_dc_converter_id="dc_dc_converter_1"), ivc)

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


def test_capacitor_capacity():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterCapacitorCapacity(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterCapacitorCapacity(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:capacitor:capacity",
            units="mF",
        )
        == pytest.approx(0.484, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_capacitor_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterCapacitorWeight(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterCapacitorWeight(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:capacitor:mass",
            units="kg",
        )
        == pytest.approx(1.21, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_module_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterCasingWeight(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterCasingWeight(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:module:mass",
            units="kg",
        )
        == pytest.approx(0.335, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_converter_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterWeight(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingDCDCConverterWeight(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass",
            units="kg",
        )
        == pytest.approx(86.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_converter_weight_by_sum():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterWeightBySum(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterWeightBySum(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass",
            units="kg",
        )
        == pytest.approx(103.545, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_converter_cg():

    expected_cg = [2.69, 0.45, 2.54]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingDCDCConverterCG(dc_dc_converter_id="dc_dc_converter_1", position=option)
            ),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingDCDCConverterCG(dc_dc_converter_id="dc_dc_converter_1", position=option), ivc
        )

        assert (
            problem.get_val(
                "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:CG:x",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True)


def test_constraints_current_capacitor_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentCapacitorEnforce(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsCurrentCapacitorEnforce(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:capacitor"
            ":current_caliber",
            units="A",
        )
        == pytest.approx(380.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_current_inductor_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentInductorEnforce(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsCurrentInductorEnforce(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor"
            ":current_caliber",
            units="A",
        )
        == pytest.approx(380.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_current_module_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentModuleEnforce(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsCurrentModuleEnforce(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:module"
            ":current_caliber",
            units="A",
        )
        == pytest.approx(390.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_current_input_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentInputEnforce(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsCurrentInputEnforce(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:current_in_caliber",
            units="A",
        )
        == pytest.approx(400.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_voltage_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnforce(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsVoltageEnforce(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_caliber",
            units="V",
        )
        == pytest.approx(860.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_voltage_input_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageInputEnforce(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsVoltageInputEnforce(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_in_caliber",
            units="V",
        )
        == pytest.approx(860.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_frequency_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsFrequencyEnforce(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsFrequencyEnforce(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:switching_frequency",
            units="Hz",
        )
        == pytest.approx(12.0e3, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_air_gap_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsInductorAirGapEnforce(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsInductorAirGapEnforce(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:air_gap",
            units="m",
        )
        == pytest.approx(0.0126775, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_current_capacitor_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentCapacitorEnsure(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsCurrentCapacitorEnsure(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:capacitor"
            ":current_caliber",
            units="A",
        )
        == pytest.approx(-20.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_current_inductor_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentInductorEnsure(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsCurrentInductorEnsure(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor"
            ":current_caliber",
            units="A",
        )
        == pytest.approx(-20.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_current_module_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentModuleEnsure(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsCurrentModuleEnsure(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:module"
            ":current_caliber",
            units="A",
        )
        == pytest.approx(-10.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_current_input_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentInputEnsure(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsCurrentInputEnsure(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:current_in_caliber",
            units="A",
        )
        == pytest.approx(0.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_voltage_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnsure(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsVoltageEnsure(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_caliber",
            units="V",
        )
        == pytest.approx(0.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_voltage_input_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageInputEnsure(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsVoltageInputEnsure(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_in_caliber",
            units="V",
        )
        == pytest.approx(0.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_frequency_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsFrequencyEnsure(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsFrequencyEnsure(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:switching_frequency",
            units="Hz",
        )
        == pytest.approx(0.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_air_gap_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsInductorAirGapEnsure(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsInductorAirGapEnsure(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor"
            ":air_gap",
            units="m",
        )
        == pytest.approx(-0.0026775, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_dc_dc_converter_sizing():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverter(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingDCDCConverter(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass",
            units="kg",
        )
        == pytest.approx(96.63, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:capacitor:capacity",
            units="mF",
        )
        == pytest.approx(0.460, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:diode:resistance",
            units="ohm",
        )
        == pytest.approx(0.002157, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:CG:x",
            units="m",
        )
        == pytest.approx(2.69, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:low_speed:CD0",
        )
        == pytest.approx(0.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:cruise:CD0",
        )
        == pytest.approx(0.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_switching_frequency_mission():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:switching_frequency_mission",
        val=12.0e3,
        units="Hz",
    )

    problem = run_system(
        PerformancesSwitchingFrequencyMission(
            dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("switching_frequency", units="Hz") == pytest.approx(
        np.full(NB_POINTS_TEST, 12.0e3), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:switching_frequency_mission",
        val=[15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 420],
        units="Hz",
    )

    problem3 = run_system(
        PerformancesSwitchingFrequencyMission(
            dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc3,
    )

    assert problem3.get_val("switching_frequency", units="Hz") == pytest.approx(
        np.array([15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 420]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_voltage_out_target_mission():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1"
        ":voltage_out_target_mission",
        val=850.0,
        units="V",
    )

    problem = run_system(
        PerformancesVoltageOutTargetMission(
            dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("voltage_out_target", units="V") == pytest.approx(
        np.full(NB_POINTS_TEST, 850), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_out_target_mission",
        val=[850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 420.0],
        units="V",
    )

    problem3 = run_system(
        PerformancesVoltageOutTargetMission(
            dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc3,
    )

    assert problem3.get_val("voltage_out_target", units="V") == pytest.approx(
        np.array([850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 420.0]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_load_side():

    ivc = om.IndepVarComp()
    power = np.linspace(350, 400, NB_POINTS_TEST)
    ivc.add_output("power", power, units="kW")
    voltage_in = np.full(NB_POINTS_TEST, 810)
    ivc.add_output("dc_voltage_in", voltage_in, units="V")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesConverterLoadSide(number_of_points=NB_POINTS_TEST), ivc)

    expected_current = np.array(
        [432.1, 439.0, 445.8, 452.7, 459.5, 466.4, 473.3, 480.1, 487.0, 493.8]
    )
    assert problem.get_val("dc_current_in", units="A") == pytest.approx(expected_current, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_duty_cycle():

    ivc = om.IndepVarComp()
    voltage_out = np.linspace(710, 910, NB_POINTS_TEST)
    ivc.add_output("dc_voltage_out", voltage_out, units="V")
    voltage_in = np.full(NB_POINTS_TEST, 810)
    ivc.add_output("dc_voltage_in", voltage_in, units="V")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesDutyCycle(number_of_points=NB_POINTS_TEST), ivc)

    expected_duty_cycle = np.array(
        [0.467, 0.475, 0.482, 0.489, 0.497, 0.503, 0.51, 0.517, 0.523, 0.529]
    )
    assert problem.get_val("duty_cycle") == pytest.approx(expected_duty_cycle, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_currents():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesCurrents(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    duty_cycle = np.array([0.467, 0.475, 0.482, 0.489, 0.497, 0.503, 0.51, 0.517, 0.523, 0.529])
    ivc.add_output("duty_cycle", duty_cycle)
    dc_current_out = np.linspace(200.0, 400.0, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCurrents(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_current_igbt = np.array(
        [256.4, 291.7, 327.6, 364.9, 404.9, 444.0, 485.8, 529.3, 572.8, 617.7]
    )
    assert problem.get_val("current_IGBT") == pytest.approx(expected_current_igbt, rel=1e-2)
    expected_current_diode = np.array(
        [273.9, 306.7, 339.6, 373.0, 407.3, 441.3, 476.2, 511.6, 547.0, 582.8]
    )
    assert problem.get_val("current_diode") == pytest.approx(expected_current_diode, rel=1e-2)
    expected_current_capacitor = np.array(
        [187.2, 211.4, 235.8, 260.9, 287.2, 313.0, 340.1, 367.9, 395.6, 423.9]
    )
    assert problem.get_val("current_capacitor") == pytest.approx(
        expected_current_capacitor, rel=1e-2
    )
    expected_current_inductor = np.array(
        [375.2, 423.3, 471.9, 521.9, 574.3, 626.0, 680.3, 736.1, 792.0, 849.3]
    )
    assert problem.get_val("current_inductor") == pytest.approx(expected_current_inductor, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_conduction_losses():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesConductionLosses(
                dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    dc_current_out = np.linspace(200.0, 400.0, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    ivc.add_output(
        "current_IGBT",
        np.array([256.4, 291.7, 327.6, 364.9, 404.9, 444.0, 485.8, 529.3, 572.8, 617.7]),
        units="A",
    )
    ivc.add_output(
        "current_diode",
        np.array([273.9, 306.7, 339.6, 373.0, 407.3, 441.3, 476.2, 511.6, 547.0, 582.8]),
        units="A",
    )
    ivc.add_output(
        "current_capacitor",
        np.array([187.2, 211.4, 235.8, 260.9, 287.2, 313.0, 340.1, 367.9, 395.6, 423.9]),
        units="A",
    )
    ivc.add_output(
        "current_inductor",
        np.array([375.2, 423.3, 471.9, 521.9, 574.3, 626.0, 680.3, 736.1, 792.0, 849.3]),
        units="A",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesConductionLosses(
            dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    expected_losses_igbt = np.array(
        [148.9, 192.8, 243.1, 301.6, 371.3, 446.4, 534.6, 634.6, 743.0, 864.2]
    )
    assert problem.get_val("conduction_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array(
        [470.5, 552.7, 641.3, 737.0, 841.0, 950.7, 1069.4, 1196.4, 1330.4, 1472.9]
    )
    assert problem.get_val("conduction_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )
    expected_losses_inductor = np.array(
        [197.1, 250.8, 311.8, 381.3, 461.8, 548.6, 647.9, 758.7, 878.1, 1009.7]
    )
    assert problem.get_val("conduction_losses_inductor", units="W") == pytest.approx(
        expected_losses_inductor, rel=1e-2
    )
    expected_losses_capacitor = np.array(
        [49.1, 62.6, 77.8, 95.3, 115.4, 137.1, 161.9, 189.4, 219.1, 251.6]
    )
    assert problem.get_val("conduction_losses_capacitor", units="W") == pytest.approx(
        expected_losses_capacitor, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_switching_losses():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesSwitchingLosses(
                dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "current_IGBT",
        np.array([256.4, 291.7, 327.6, 364.9, 404.9, 444.0, 485.8, 529.3, 572.8, 617.7]),
        units="A",
    )
    ivc.add_output(
        "current_diode",
        np.array([273.9, 306.7, 339.6, 373.0, 407.3, 441.3, 476.2, 511.6, 547.0, 582.8]),
        units="A",
    )
    ivc.add_output("switching_frequency", np.linspace(3000.0, 12000.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesSwitchingLosses(
            dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    expected_losses_igbt = np.array(
        [151.1, 218.7, 296.1, 384.6, 486.5, 599.6, 728.7, 874.4, 1035.5, 1215.1]
    )
    assert problem.get_val("switching_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array(
        [96.8, 142.7, 195.4, 255.1, 322.1, 395.9, 477.2, 565.9, 661.5, 764.5]
    )
    assert problem.get_val("switching_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_total_losses_dc_dc_converter():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "switching_losses_IGBT",
        [131.3, 185.3, 244.7, 309.6, 380.4, 457.0, 539.7, 628.6, 724.0, 825.9],
        units="W",
    )
    ivc.add_output(
        "switching_losses_diode",
        [73.6, 107.5, 146.0, 189.1, 236.8, 289.0, 345.8, 407.2, 473.0, 543.3],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_IGBT",
        [148.9, 192.8, 243.1, 301.6, 371.3, 446.4, 534.6, 634.6, 743.0, 864.2],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_diode",
        [470.5, 552.7, 641.3, 737.0, 841.0, 950.7, 1069.4, 1196.4, 1330.4, 1472.9],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_inductor",
        [197.1, 250.8, 311.8, 381.3, 461.8, 548.6, 647.9, 758.7, 878.1, 1009.7],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_capacitor",
        [49.1, 62.6, 77.8, 95.3, 115.4, 137.1, 161.9, 189.4, 219.1, 251.6],
        units="W",
    )

    problem = run_system(PerformancesLosses(number_of_points=NB_POINTS_TEST), ivc)

    expected_losses = np.array(
        [1070.5, 1351.7, 1664.7, 2013.9, 2406.7, 2828.8, 3299.3, 3814.9, 4367.6, 4967.6]
    )
    assert problem.get_val("losses_converter", units="W") == pytest.approx(
        expected_losses, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_converter_efficiency():

    ivc = om.IndepVarComp()
    voltage_out = np.linspace(710, 910, NB_POINTS_TEST)
    ivc.add_output("dc_voltage_out", voltage_out, units="V")
    dc_current_out = np.linspace(200.0, 400.0, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    total_losses = np.array(
        [1070.5, 1351.7, 1664.7, 2013.9, 2406.7, 2828.8, 3299.3, 3814.9, 4367.6, 4967.6]
    )
    ivc.add_output("losses_converter", val=total_losses, units="W")

    # Won't be representative since the modulation index used for the computation of the losses
    # is not equal to the modulation index computed based on bus voltage
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEfficiency(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_efficiency = np.array(
        [0.993, 0.992, 0.991, 0.99, 0.99, 0.989, 0.988, 0.988, 0.987, 0.987]
    )
    assert problem.get_val("efficiency") == pytest.approx(expected_efficiency, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "dc_current_in",
        np.array([432.1, 439.0, 445.8, 452.7, 459.5, 466.4, 473.3, 480.1, 487.0, 493.8]),
        units="A",
    )
    ivc.add_output(
        "current_IGBT",
        np.array([256.4, 291.7, 327.6, 364.9, 404.9, 444.0, 485.8, 529.3, 572.8, 617.7]),
        units="A",
    )
    ivc.add_output(
        "current_diode",
        np.array([273.9, 306.7, 339.6, 373.0, 407.3, 441.3, 476.2, 511.6, 547.0, 582.8]),
        units="A",
    )
    ivc.add_output(
        "current_capacitor",
        np.array([187.2, 211.4, 235.8, 260.9, 287.2, 313.0, 340.1, 367.9, 395.6, 423.9]),
        units="A",
    )
    ivc.add_output(
        "current_inductor",
        np.array([375.2, 423.3, 471.9, 521.9, 574.3, 626.0, 680.3, 736.1, 792.0, 849.3]),
        units="A",
    )
    ivc.add_output("switching_frequency", np.linspace(3000.0, 12000.0, NB_POINTS_TEST))
    voltage_out = np.linspace(710, 910, NB_POINTS_TEST)
    ivc.add_output("dc_voltage_out", voltage_out, units="V")
    voltage_in = np.full(NB_POINTS_TEST, 810)
    ivc.add_output("dc_voltage_in", voltage_in, units="V")
    total_losses = np.array(
        [1070.5, 1351.7, 1664.7, 2013.9, 2406.7, 2828.8, 3299.3, 3814.9, 4367.6, 4967.6]
    )
    ivc.add_output("losses_converter", val=total_losses, units="W")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(
            dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:current_in_max",
            units="A",
        )
        == pytest.approx(493.8, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:igbt:current_max",
            units="A",
        )
        == pytest.approx(617.7, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:diode:current_max",
            units="A",
        )
        == pytest.approx(582.8, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:current_max",
            units="A",
        )
        == pytest.approx(849.3, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:capacitor:current_max",
            units="A",
        )
        == pytest.approx(423.9, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_in_max",
            units="V",
        )
        == pytest.approx(810.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_out_max",
            units="V",
        )
        == pytest.approx(910.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:losses_max",
            units="W",
        )
        == pytest.approx(4967.6, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:switching_frequency_max",
            units="Hz",
        )
        == pytest.approx(12000, rel=1e-2)
    )

    problem.check_partials(compact_print=True)
