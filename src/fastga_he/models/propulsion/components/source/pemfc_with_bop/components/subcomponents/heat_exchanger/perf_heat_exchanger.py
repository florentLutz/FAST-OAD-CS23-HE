# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_coolant_heat_capacity import PerformancesCoolantHeatCapacity
from .perf_air_heat_capacity import PerformancesAirHeatCapacity
from .perf_mean_air_temperature import PerformancesMeanAirTemperature
from .perf_heat_capacity_ratio import PerformancesHeatCapacityRatio
from .perf_heat_exchanger_NTU import PerformancesHeatExchangerNTU
from .perf_heat_exchanger_UA import PerformancesHeatExchangerUA
from .perf_air_mass_velocity import PerformancesAirMassVelocity
from .perf_coolant_mass_velocity import PerformancesCoolantMassVelocity
from .perf_air_reynold_number import PerformancesAirReynoldsNumber
from .perf_coolant_reynold_number import PerformancesCoolantReynoldsNumber
from .perf_fanning_friction_factor import PerformancesFanningFrictionFactor
from .perf_pressure_drop_coefficient import PerformancesPressureDropCoefficient
from .perf_air_pressure_drop import PerformancesAirPressureDrop
from .perf_coolant_pressure_drop import PerformancesCoolantPressureDrop

from ..fluid_characteristics import (
    FluidDensity,
    FluidSpecificHeatCapacity,
    FluidThermalConductivity,
    FluidDynamicViscosity,
    FluidPrandtlNumber,
)


class PerformancesHeatExchanger(om.Group):
    """
    Performances of the heat exchanger.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )
        self.options.declare(
            "coolant_fluid_type",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        fluid = self.options["coolant_fluid_type"]
        heat_exchanger_id = self.options["heat_exchanger_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "mean_coolant_density_hex_performances",
            FluidDensity(fluid=fluid),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:mean_temperature",
                ),
                (
                    "fluid_density",
                    "mean_coolant_density",
                ),
            ],
        )
        self.add_subsystem(
            "mean_coolant_specific_heat_capacity_hex_performances",
            FluidSpecificHeatCapacity(fluid=fluid),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:mean_temperature",
                ),
                ("fluid_specific_heat_capacity", "mean_coolant_specific_heat_capacity"),
            ],
        )
        self.add_subsystem(
            "mean_coolant_thermal_conductivity_hex_performances",
            FluidThermalConductivity(fluid=fluid),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:mean_temperature",
                ),
                (
                    "fluid_thermal_conductivity",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_coolant_thermal_conductivity",
                ),
            ],
        )
        self.add_subsystem(
            "mean_coolant_dynamic_viscosity_hex_performances",
            FluidDynamicViscosity(fluid=fluid),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:mean_temperature",
                ),
                (
                    "fluid_dynamic_viscosity",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_coolant_dynamic_viscosity",
                ),
            ],
        )
        self.add_subsystem(
            "mean_coolant_prandtl_number_hex_performances",
            FluidPrandtlNumber(fluid=fluid),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:mean_temperature",
                ),
                (
                    "fluid_prandtl_number",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_coolant_prandtl_number",
                ),
            ],
        )
        self.add_subsystem(
            "coolant_heat_capacity_hex_performances",
            PerformancesCoolantHeatCapacity(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "mean_air_temperature_hex_performances",
            PerformancesMeanAirTemperature(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "mean_air_density_hex_performances",
            FluidDensity(fluid="air"),
            promotes=[
                ("fluid_pressure", "air_static_pressure"),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_air_temperature",
                ),
                ("fluid_density", "mean_air_density"),
            ],
        )
        self.add_subsystem(
            "mean_air_specific_heat_capacity_hex_performances",
            FluidSpecificHeatCapacity(fluid="air"),
            promotes=[
                ("fluid_pressure", "air_static_pressure"),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_air_temperature",
                ),
                ("fluid_specific_heat_capacity", "mean_air_specific_heat_capacity"),
            ],
        )
        self.add_subsystem(
            "mean_air_thermal_conductivity_hex_performances",
            FluidThermalConductivity(fluid="air"),
            promotes=[
                ("fluid_pressure", "air_static_pressure"),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_air_temperature",
                ),
                (
                    "fluid_thermal_conductivity",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_air_thermal_conductivity",
                ),
            ],
        )
        self.add_subsystem(
            "mean_air_dynamic_viscosity_hex_performances",
            FluidDynamicViscosity(fluid="air"),
            promotes=[
                ("fluid_pressure", "air_static_pressure"),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_air_temperature",
                ),
                (
                    "fluid_dynamic_viscosity",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_air_dynamic_viscosity",
                ),
            ],
        )
        self.add_subsystem(
            "mean_air_prandtl_number_hex_performances",
            FluidPrandtlNumber(fluid="air"),
            promotes=[
                ("fluid_pressure", "air_static_pressure"),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_air_temperature",
                ),
                (
                    "fluid_prandtl_number",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + heat_exchanger_id
                    + ":mean_air_prandtl_number",
                ),
            ],
        )
        self.add_subsystem(
            "air_heat_capacity_hex_performances",
            PerformancesAirHeatCapacity(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_capacity_ratio",
            PerformancesHeatCapacityRatio(),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_exchanger_NTU",
            PerformancesHeatExchangerNTU(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_exchanger_UA",
            PerformancesHeatExchangerUA(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_mass_velocity",
            PerformancesAirMassVelocity(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "coolant_mass_velocity",
            PerformancesCoolantMassVelocity(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_reynolds_number",
            PerformancesAirReynoldsNumber(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "coolant_reynolds_number",
            PerformancesCoolantReynoldsNumber(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "fanning_friction_factor",
            PerformancesFanningFrictionFactor(),
            promotes=["*"],
        )
        self.add_subsystem(
            "pressure_drop_coefficient",
            PerformancesPressureDropCoefficient(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_inlet_density_hex_performances",
            FluidDensity(fluid="air"),
            promotes=[
                ("fluid_pressure", "air_static_pressure"),
                ("fluid_temperature", "air_inlet_temperature"),
                ("fluid_density", "air_inlet_density"),
            ],
        )
        self.add_subsystem(
            "air_outlet_density_hex_performances",
            FluidDensity(fluid="air"),
            promotes=[
                ("fluid_pressure", "air_static_pressure"),
                ("fluid_temperature", "air_outlet_temperature"),
                ("fluid_density", "air_outlet_density"),
            ],
        )
        self.add_subsystem(
            "air_pressure_drop",
            PerformancesAirPressureDrop(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                heat_exchanger_id=heat_exchanger_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "coolant_inlet_density_hex_performances",
            FluidDensity(fluid=fluid),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:static_pressure",
                ),
                ("fluid_temperature", "coolant_inlet_temperature"),
                ("fluid_density", "coolant_inlet_density"),
            ],
        )
        self.add_subsystem(
            "coolant_outlet_density_hex_performances",
            FluidDensity(fluid=fluid),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:static_pressure",
                ),
                ("fluid_temperature", "coolant_outlet_temperature"),
                ("fluid_density", "coolant_outlet_density"),
            ],
        )
        self.add_subsystem(
            "coolant_pressure_drop",
            PerformancesCoolantPressureDrop(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 10
        self.nonlinear_solver.options["rtol"] = 1e-4
        self.nonlinear_solver.options["stall_limit"] = 7
        self.nonlinear_solver.options["stall_tol"] = 1e-5
        self.linear_solver = om.DirectSolver()
