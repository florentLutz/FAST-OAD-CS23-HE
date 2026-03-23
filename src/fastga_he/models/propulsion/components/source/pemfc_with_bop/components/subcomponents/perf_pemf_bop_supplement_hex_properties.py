# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .fluid_characteristics import FluidSpecificHeatCapacity


class PerformancesSupplementHeatExchangerThermalBalance(om.Group):
    """
    Group to compute the thermal properties of the supplement heat exchanger.
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
            "coolant_fluid_type",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )
        self.options.declare(
            name="primary_heat_exchanger_id",
            default=None,
            desc="Identifier of the primary heat exchanger",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_fluid_type = self.options["coolant_fluid_type"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        self.add_subsystem(
            "supplement_heat_exchanger_air_properties",
            _SupplementHeatExchangerAirProperties(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "mean_air_specific_heat_capacity_hex_performances",
            FluidSpecificHeatCapacity(fluid="air"),
            promotes=[
                ("fluid_pressure", "air_static_pressure"),
                ("fluid_temperature", "mean_air_temperature"),
                ("fluid_specific_heat_capacity", "mean_air_specific_heat_capacity"),
            ],
        )
        self.add_subsystem(
            "mean_coolant_specific_heat_capacity_hex_performances",
            FluidSpecificHeatCapacity(fluid=coolant_fluid_type),
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
            "air_outlet_temperature_hex_performances",
            _AirOutletTemperate(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                primary_heat_exchanger_id=primary_heat_exchanger_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "coolant_temperature_hex_performances",
            _CoolantTemperature(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                primary_heat_exchanger_id=primary_heat_exchanger_id,
            ),
            promotes=["*"],
        )


class _SupplementHeatExchangerAirProperties(om.ExplicitComponent):
    """
    Compute the specific heat capacity of the air at the primary heat exchanger inlet and outlet.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "diffuser_exit_total_temperature",
            val=np.nan,
            units="K",
            shape=number_of_points,
        )
        self.add_input(
            "diffuser_exit_total_pressure",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )
        self.add_input(
            "exterior_temperature",
            val=np.nan,
            units="K",
            shape=number_of_points,
        )
        self.add_input(
            "ambient_pressure",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )
        self.add_input(
            "total_air_mass_flow_rate",
            val=np.nan,
            units="kg/s",
            shape=number_of_points,
        )

        self.add_input("air_inlet_temperature", val=np.nan, units="K")
        self.add_output("mean_air_temperature", val=np.nan, units="K")
        self.add_output("air_static_pressure", val=np.nan, units="Pa")
        self.add_output("max_total_air_flow_rate", val=np.nan, units="kg/s")

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="mean_air_temperature",
            wrt=["diffuser_exit_total_temperature", "exterior_temperature"],
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="air_static_pressure",
            wrt=["diffuser_exit_total_pressure", "ambient_pressure"],
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="inlet_air_temperature",
            wrt="exterior_temperature",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="max_total_air_flow_rate",
            wrt="total_air_mass_flow_rate",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        diffuser_exit_total_temperature = inputs["diffuser_exit_total_temperature"]
        exterior_temperature = inputs["exterior_temperature"]
        diffuser_exit_total_pressure = inputs["diffuser_exit_total_pressure"]
        ambient_pressure = inputs["ambient_pressure"]
        total_air_mass_flow_rate = inputs["total_air_mass_flow_rate"]

        outputs["mean_air_temperature"] = (
            np.mean(diffuser_exit_total_temperature + exterior_temperature) / 2.0
        )
        outputs["air_static_pressure"] = (
            np.mean(diffuser_exit_total_pressure + ambient_pressure) / 2.0
        )
        outputs["inlet_air_temperature"] = np.max(exterior_temperature)
        outputs["max_total_air_flow_rate"] = np.max(total_air_mass_flow_rate)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        diffuser_exit_total_temperature = inputs["diffuser_exit_total_temperature"]
        exterior_temperature = inputs["exterior_temperature"]
        diffuser_exit_total_pressure = inputs["diffuser_exit_total_pressure"]
        ambient_pressure = inputs["ambient_pressure"]
        total_air_mass_flow_rate = inputs["total_air_mass_flow_rate"]

        mean_air_temperature = np.mean(diffuser_exit_total_temperature + exterior_temperature) / 2.0
        static_air_pressure = np.mean(diffuser_exit_total_pressure + ambient_pressure) / 2.0
        air_inlet_temperature = np.max(exterior_temperature)
        max_air_flow_rate = np.max(total_air_mass_flow_rate)

        partials["mean_air_temperature", "diffuser_exit_total_temperature"] = np.where(
            mean_air_temperature == (diffuser_exit_total_temperature + exterior_temperature) / 2.0,
            0.5,
            0.0,
        )

        partials["mean_air_temperature", "exterior_temperature"] = np.where(
            mean_air_temperature == (diffuser_exit_total_temperature + exterior_temperature) / 2.0,
            0.5,
            0.0,
        )

        partials["air_static_pressure", "diffuser_exit_total_pressure"] = np.where(
            static_air_pressure == (diffuser_exit_total_pressure + ambient_pressure) / 2.0, 0.5, 0.0
        )

        partials["air_static_pressure", "ambient_pressure"] = np.where(
            static_air_pressure == (diffuser_exit_total_pressure + ambient_pressure) / 2.0, 0.5, 0.0
        )

        partials["inlet_air_temperature", "exterior_temperature"] = np.where(
            air_inlet_temperature == exterior_temperature, 1.0, 0.0
        )

        partials["max_total_air_flow_rate", "total_air_mass_flow_rate"] = np.where(
            max_air_flow_rate == total_air_mass_flow_rate, 1.0, 0.0
        )


class _SupplementAirFlowRate(om.ExplicitComponent):
    """
    Compute the air mass flow rate at the supplement heat exchanger inlet.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
            units="kg/s",
            val=np.nan,
        )
        self.add_input(
            "max_total_air_flow_rate",
            val=np.nan,
            units="kg/s",
        )

        self.add_output(
            "air_mass_flow_rate",
            val=np.nan,
            units="kg/s",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", val=-1.0)
        self.declare_partials("*", "max_total_air_flow_rate", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs["air_mass_flow_rate"] = (
            inputs["max_total_air_flow_rate"]
            - inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_consumption_max"
            ]
        )


class _MinimumHeatCapacity(om.ExplicitComponent):
    """
    Compute the minimum heat capacity of the primary heat exchanger.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            "air_mass_flow_rate",
            val=np.nan,
            units="kg/s",
        )
        self.add_input(
            "air_mean_specific_heat_capacity",
            val=np.nan,
            units="J/kg/K",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            val=np.nan,
            units="kg/s",
        )
        self.add_input(
            name="mean_coolant_specific_heat_capacity",
            val=np.nan,
            units="J/kg/K",
        )

        self.add_output(
            name="minimum_heat_capacity",
            val=np.nan,
            units="W/K",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        air_mass_flow_rate = inputs["air_mass_flow_rate"]
        air_mean_specific_heat_capacity = inputs["air_mean_specific_heat_capacity"]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        mean_coolant_specific_heat_capacity = inputs["mean_coolant_specific_heat_capacity"]

        outputs["minimum_heat_capacity"] = min(
            air_mass_flow_rate * air_mean_specific_heat_capacity,
            coolant_mass_flow_rate * mean_coolant_specific_heat_capacity,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        air_mass_flow_rate = inputs["air_mass_flow_rate"]
        air_mean_specific_heat_capacity = inputs["air_mean_specific_heat_capacity"]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        mean_coolant_specific_heat_capacity = inputs["mean_coolant_specific_heat_capacity"]
        min_heat_capacity = min(
            air_mass_flow_rate * air_mean_specific_heat_capacity,
            coolant_mass_flow_rate * mean_coolant_specific_heat_capacity,
        )

        if air_mass_flow_rate * air_mean_specific_heat_capacity == min_heat_capacity:
            partials["minimum_heat_capacity", "air_mass_flow_rate"] = (
                air_mean_specific_heat_capacity
            )
            partials["minimum_heat_capacity", "air_mean_specific_heat_capacity"] = (
                air_mass_flow_rate
            )
            partials[
                "minimum_heat_capacity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":coolant:mass_flow_rate",
            ] = 0.0
            partials["minimum_heat_capacity", "mean_coolant_specific_heat_capacity"] = 0.0

        else:
            partials["minimum_heat_capacity", "air_mass_flow_rate"] = 0.0
            partials["minimum_heat_capacity", "air_mean_specific_heat_capacity"] = 0.0
            partials[
                "minimum_heat_capacity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":coolant:mass_flow_rate",
            ] = mean_coolant_specific_heat_capacity
            partials["minimum_heat_capacity", "mean_coolant_specific_heat_capacity"] = (
                coolant_mass_flow_rate
            )


class _AirOutletTemperate(om.ExplicitComponent):
    """
    Compute the outlet temperature of the air at the primary heat exchanger outlet.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="primary_heat_exchanger_id",
            default=None,
            desc="Identifier of the primary heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        self.add_input(
            "air_inlet_temperature",
            val=np.nan,
            units="K",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:outlet_temperature",
            val=np.nan,
            units="K",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
            units="kg/s",
            val=np.nan,
        )
        self.add_input(
            "mean_air_specific_heat_capacity",
            val=np.nan,
            units="J/kg/K",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":heat_exchanger_efficiency",
            val=0.98,
            units="unitless",
        )
        self.add_input(
            name="minimum_heat_capacity",
            val=np.nan,
            units="W/K",
        )

        self.add_output(
            "air_outlet_temperature",
            val=np.nan,
            units="K",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        air_inlet_temperature = inputs["air_inlet_temperature"]
        coolant_outlet_temperature = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:outlet_temperature"
        ]
        air_consumption_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]
        air_mean_specific_heat_capacity = inputs["air_mean_specific_heat_capacity"]
        mini_heat_capacity = inputs["minimum_heat_capacity"]
        heat_exchanger_efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":heat_exchanger_efficiency"
        ]

        outputs["air_outlet_temperature"] = (
            air_inlet_temperature
            + heat_exchanger_efficiency
            * mini_heat_capacity
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (air_consumption_max * air_mean_specific_heat_capacity)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        air_inlet_temperature = inputs["air_inlet_temperature"]
        coolant_outlet_temperature = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:outlet_temperature"
        ]
        air_consumption_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]
        air_mean_specific_heat_capacity = inputs["air_mean_specific_heat_capacity"]
        mini_heat_capacity = inputs["minimum_heat_capacity"]
        heat_exchanger_efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":heat_exchanger_efficiency"
        ]

        partials["air_outlet_temperature", "air_inlet_temperature"] = (
            1.0
            - heat_exchanger_efficiency
            * mini_heat_capacity
            / (air_consumption_max * air_mean_specific_heat_capacity)
        )

        partials[
            "air_outlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:outlet_temperature",
        ] = (
            heat_exchanger_efficiency
            * mini_heat_capacity
            / (air_consumption_max * air_mean_specific_heat_capacity)
        )

        partials[
            "air_outlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
        ] = (
            -heat_exchanger_efficiency
            * mini_heat_capacity
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (air_consumption_max**2.0 * air_mean_specific_heat_capacity)
        )

        partials["air_outlet_temperature", "air_mean_specific_heat_capacity"] = (
            -heat_exchanger_efficiency
            * mini_heat_capacity
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (air_consumption_max * air_mean_specific_heat_capacity**2.0)
        )

        partials["air_outlet_temperature", "minimum_heat_capacity"] = (
            heat_exchanger_efficiency
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (air_consumption_max * air_mean_specific_heat_capacity)
        )

        partials[
            "air_outlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":heat_exchanger_efficiency",
        ] = (
            mini_heat_capacity
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (air_consumption_max * air_mean_specific_heat_capacity)
        )


class _CoolantTemperature(om.ExplicitComponent):
    """
    The computation of the coolant temperature at the primary heat exchanger's inlet and outlet.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="primary_heat_exchanger_id",
            default=None,
            desc="Identifier of the primary heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:intermediate_temperature",
            val=np.nan,
            units="K",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:outlet_temperature",
            val=np.nan,
            units="K",
        )

        self.add_output(
            name="inlet_coolant_temperature",
            val=np.nan,
            units="K",
        )
        self.add_output(
            name="outlet_coolant_temperature",
            val=np.nan,
            units="K",
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        self.declare_partials(
            "outlet_coolant_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:intermediate_temperature",
            val=1.0,
        )
        self.declare_partials(
            "inlet_coolant_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:outlet_temperature",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        outputs["outlet_coolant_temperature"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:intermediate_temperature"
        ]
        outputs["inlet_coolant_temperature"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":coolant:outlet_temperature"
        ]
