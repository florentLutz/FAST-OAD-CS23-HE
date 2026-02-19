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
