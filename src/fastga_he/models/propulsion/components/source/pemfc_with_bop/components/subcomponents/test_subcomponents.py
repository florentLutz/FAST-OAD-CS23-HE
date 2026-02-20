# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import pytest
import openmdao.api as om
from tests.testing_utilities import run_system

from .fluid_characteristics.fluid_density import FluidDensity
from .fluid_characteristics.fluid_specific_heat_capacity import FluidSpecificHeatCapacity
from .fluid_characteristics.fluid_thermal_conductivity import FluidThermalConductivity
from .fluid_characteristics.fluid_dynamic_viscosity import FluidDynamicViscosity
from .fluid_characteristics.fluid_prandtl_number import FluidPrandtlNumber
from .fluid_characteristics.fluid_enthalpy import FluidEnthalpy
from .fluid_characteristics.fluid_specific_volume import FluidSpecificVolume

from .humidifier.perf_humidifier_rating_pressure_drop import (
    PerformancesHumidifierRatingPressureDrop,
)
from .humidifier.sizing_humidifier_volume import SizingHumidifierVolume
from .humidifier.sizing_humidifier_weight import SizingHumidifierWeight


def test_fluid_density():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidDensity(), ivc)
    assert problem.get_val("fluid_density", units="kg/m**3") == pytest.approx(1.177, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fluid_specific_heat_capacity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidSpecificHeatCapacity(), ivc)
    assert problem.get_val("fluid_specific_heat_capacity", units="J/kg/K") == pytest.approx(
        1006.4, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_fluid_thermal_conductivity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidThermalConductivity(), ivc)
    assert problem.get_val("fluid_thermal_conductivity", units="W/m/K") == pytest.approx(
        0.0264, rel=1e-2
    )


def test_fluid_dynamic_viscosity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidDynamicViscosity(), ivc)
    assert problem.get_val("fluid_dynamic_viscosity", units="Pa*s") == pytest.approx(
        1.85e-5, rel=1e-2
    )


def test_fluid_prandtl_number():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidPrandtlNumber(), ivc)
    assert problem.get_val("fluid_prandtl_number", units="unitless") == pytest.approx(
        0.71, rel=1e-2
    )


def test_fluid_enthalpy():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidEnthalpy(), ivc)
    assert problem.get_val("fluid_enthalpy", units="J/kg") == pytest.approx(426297.77, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fluid_specific_volume():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidSpecificVolume(), ivc)
    assert problem.get_val("fluid_specific_volume", units="m**3/kg") == pytest.approx(
        0.85, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_performances_humidifier_rating_pressure_drop():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:oxidizer_temperature",
        units="degC",
        val=35.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:operating_pressure",
        units="atm",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:air_consumption_max",
        units="kg/h",
        val=2600.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHumidifierRatingPressureDrop(pemfc_stack_bop_id="pemfc_stack_bop_1"), ivc
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1"
        ":humidifier:max_pressure_drop",
        units="MPa",
    ) == pytest.approx(0.091, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_humidifier_weight():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_rating",
        units="kW",
        val=200.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHumidifierWeight(pemfc_stack_bop_id="pemfc_stack_bop_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1" ":humidifier:mass",
        units="kg",
    ) == pytest.approx(16.33, rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_rating",
        units="kW",
        val=1200.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHumidifierWeight(pemfc_stack_bop_id="pemfc_stack_bop_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1" ":humidifier:mass",
        units="kg",
    ) == pytest.approx(97.98, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_humidifier_volume():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_rating",
        units="kW",
        val=200.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHumidifierVolume(pemfc_stack_bop_id="pemfc_stack_bop_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1" ":humidifier:volume",
        units="m**3",
    ) == pytest.approx(0.016, rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_rating",
        units="kW",
        val=1200.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHumidifierVolume(pemfc_stack_bop_id="pemfc_stack_bop_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1" ":humidifier:volume",
        units="m**3",
    ) == pytest.approx(0.124, rel=1e-2)

    problem.check_partials(compact_print=True)
