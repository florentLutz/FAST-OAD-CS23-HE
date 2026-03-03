# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_coolant_heat_capacity import PerformancesCoolantHeatCapacity
from .perf_air_heat_capacity import PerformancesAirHeatCapacity
from .perf_mean_air_temperature import PerformancesMeanAirTemperature
from .perf_heat_capacity_ratio import PerformancesHeatCapacityRatio

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
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
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
                ("fluid_density", "mean_coolant_density"),
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
                ("fluid_thermal_conductivity", "mean_coolant_thermal_conductivity"),
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
                ("fluid_dynamic_viscosity", "mean_coolant_dynamic_viscosity"),
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
                ("fluid_prandtl_number", "mean_coolant_prandtl_number"),
            ],
        )
        self.add_subsystem(
            "coolant_heat_capacity_hex_performances",
            PerformancesCoolantHeatCapacity(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "mean_air_temperature_hex_performances",
            PerformancesMeanAirTemperature(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "mean_air_density_hex_performances",
            FluidDensity(fluid="air"),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:air_static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_air_temperature",
                ),
                ("fluid_density", "mean_air_density"),
            ],
        )
        self.add_subsystem(
            "mean_air_specific_heat_capacity_hex_performances",
            FluidSpecificHeatCapacity(fluid="air"),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:air_static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_air_temperature",
                ),
                ("fluid_specific_heat_capacity", "mean_air_specific_heat_capacity"),
            ],
        )
        self.add_subsystem(
            "mean_air_thermal_conductivity_hex_performances",
            FluidThermalConductivity(fluid="air"),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:air_static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_air_temperature",
                ),
                ("fluid_thermal_conductivity", "mean_air_thermal_conductivity"),
            ],
        )
        self.add_subsystem(
            "mean_air_dynamic_viscosity_hex_performances",
            FluidDynamicViscosity(fluid="air"),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:air_static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_air_temperature",
                ),
                ("fluid_dynamic_viscosity", "mean_air_dynamic_viscosity"),
            ],
        )
        self.add_subsystem(
            "mean_air_prandtl_number_hex_performances",
            FluidPrandtlNumber(fluid="air"),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:air_static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_air_temperature",
                ),
                ("fluid_prandtl_number", "mean_air_prandtl_number"),
            ],
        )
        self.add_subsystem(
            "air_heat_capacity_hex_performances",
            PerformancesAirHeatCapacity(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_capacity_ratio",
            PerformancesHeatCapacityRatio(),
            promotes=["*"],
        )
