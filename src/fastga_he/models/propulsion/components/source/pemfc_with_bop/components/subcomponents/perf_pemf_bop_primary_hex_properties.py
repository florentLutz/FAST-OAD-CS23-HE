# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .fluid_characteristics import FluidSpecificHeatCapacity


class PerformancesPrimaryHeatExchangerThermalBalance(om.Group):
    """
    Group to compute the thermal properties of the primary heat exchanger.
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
            "primary_heat_exchanger_air_properties",
            _PrimaryHeatExchangerAirProperties(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "primary_heat_exchanger_mean_air_temperature",
            _PrimaryHeatExchangerMeanAirTemperature(),
            promotes=["*"],
        )
        self.add_subsystem(
            "mean_air_specific_heat_capacity_hex_performances",
            FluidSpecificHeatCapacity(fluid="air"),
            promotes=[
                ("fluid_pressure", "air_static_pressure"),
                ("fluid_temperature", "mean_air_temperature"),
                ("fluid_specific_heat_capacity", "air_mean_specific_heat_capacity"),
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
            "minimum_heat_capacity",
            _MinimumHeatCapacity(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "coolant_intermediate_temperature_hex_performances",
            _CoolantIntermediateTemperate(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                primary_heat_exchanger_id=primary_heat_exchanger_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "coolant_temperature_hex_performances",
            _CoolantTemperature(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_flow_rate_hex_performances",
            _AirFlowRate(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                primary_heat_exchanger_id=primary_heat_exchanger_id,
            ),
            promotes=["*"],
        )


class _PrimaryHeatExchangerAirProperties(om.ExplicitComponent):
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
            "compressor_outlet_temperature",
            val=np.nan,
            units="K",
            shape=number_of_points,
        )
        self.add_input(
            "oxidizer_temperature",
            val=np.nan,
            units="K",
            shape=number_of_points,
        )
        self.add_input(
            "oxidizer_pressure",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )
        self.add_input(
            "compressor_pressure_supply",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )

        self.add_output("air_inlet_temperature", val=300.0, units="K")
        self.add_output("air_outlet_temperature", val=360.0, units="K")
        self.add_output("air_static_pressure", val=101325, units="Pa")

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="air_inlet_temperature",
            wrt="compressor_outlet_temperature",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="air_outlet_temperature",
            wrt="oxidizer_temperature",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="air_static_pressure",
            wrt=["compressor_pressure_supply", "oxidizer_pressure"],
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=1.0 / (2.0 * number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]

        outputs["air_inlet_temperature"] = np.max(inputs["compressor_outlet_temperature"])
        outputs["air_outlet_temperature"] = np.min(inputs["oxidizer_temperature"])
        outputs["air_static_pressure"] = np.sum(
            inputs["compressor_pressure_supply"] + inputs["oxidizer_pressure"]
        ) / (2.0 * number_of_points)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        max_inlet_temperature = np.max(inputs["compressor_outlet_temperature"])
        min_outlet_temperature = np.min(inputs["oxidizer_temperature"])

        partials["air_inlet_temperature", "compressor_outlet_temperature"] = np.where(
            inputs["compressor_outlet_temperature"] == max_inlet_temperature, 1.0, 1e-6
        )

        partials["air_outlet_temperature", "oxidizer_temperature"] = np.where(
            inputs["oxidizer_temperature"] == min_outlet_temperature, 1.0, 1e-6
        )


class _PrimaryHeatExchangerMeanAirTemperature(om.ExplicitComponent):
    """
    Compute the mean air temperature at the primary heat exchanger.
    """

    def setup(self):
        self.add_input(
            "air_inlet_temperature",
            val=np.nan,
            units="K",
        )
        self.add_input(
            "air_outlet_temperature",
            val=np.nan,
            units="K",
        )

        self.add_output(
            "mean_air_temperature",
            val=330.0,
            units="K",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", val=0.5)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["mean_air_temperature"] = (
            inputs["air_inlet_temperature"] + inputs["air_outlet_temperature"]
        ) / 2.0


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
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
            units="kg/s",
            val=np.nan,
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
            val=300.0,
            units="W/K",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        air_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]
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

        air_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]
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
            partials[
                "minimum_heat_capacity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_consumption_max",
            ] = air_mean_specific_heat_capacity
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
            partials[
                "minimum_heat_capacity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_consumption_max",
            ] = 0.0
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


class _CoolantIntermediateTemperate(om.ExplicitComponent):
    """
    Compute the thermal balance of the primary heat exchanger.
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
            "air_outlet_temperature",
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
            "air_mean_specific_heat_capacity",
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
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            val=331.0,
            units="K",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        air_inlet_temperature = inputs["air_inlet_temperature"]
        air_outlet_temperature = inputs["air_outlet_temperature"]
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

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature"
        ] = air_inlet_temperature + air_consumption_max * air_mean_specific_heat_capacity * (
            air_outlet_temperature - air_inlet_temperature
        ) / (mini_heat_capacity * heat_exchanger_efficiency)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        air_inlet_temperature = inputs["air_inlet_temperature"]
        air_outlet_temperature = inputs["air_outlet_temperature"]
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

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            "air_inlet_temperature",
        ] = 1.0 - (
            air_consumption_max
            * air_mean_specific_heat_capacity
            / (mini_heat_capacity * heat_exchanger_efficiency)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            "air_outlet_temperature",
        ] = (
            air_consumption_max
            * air_mean_specific_heat_capacity
            / (mini_heat_capacity * heat_exchanger_efficiency)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
        ] = (
            air_mean_specific_heat_capacity
            * (air_outlet_temperature - air_inlet_temperature)
            / (mini_heat_capacity * heat_exchanger_efficiency)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            "air_mean_specific_heat_capacity",
        ] = (
            air_consumption_max
            * (air_outlet_temperature - air_inlet_temperature)
            / (mini_heat_capacity * heat_exchanger_efficiency)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            "minimum_heat_capacity",
        ] = -(
            air_consumption_max
            * air_mean_specific_heat_capacity
            * (air_outlet_temperature - air_inlet_temperature)
            / (mini_heat_capacity**2.0 * heat_exchanger_efficiency)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":heat_exchanger_efficiency",
        ] = -(
            air_consumption_max
            * air_mean_specific_heat_capacity
            * (air_outlet_temperature - air_inlet_temperature)
            / (mini_heat_capacity * heat_exchanger_efficiency**2.0)
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

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            val=np.nan,
            units="K",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:inlet_temperature",
            val=np.nan,
            units="K",
        )

        self.add_output(
            name="coolant_inlet_temperature",
            val=360.0,
            units="K",
        )
        self.add_output(
            name="coolant_outlet_temperature",
            val=330.0,
            units="K",
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            "coolant_inlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            val=1.0,
        )
        self.declare_partials(
            "coolant_outlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:inlet_temperature",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs["coolant_inlet_temperature"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature"
        ]
        outputs["coolant_outlet_temperature"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:inlet_temperature"
        ]


class _AirFlowRate(om.ExplicitComponent):
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
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
            units="kg/s",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":air_flow_rate",
            units="kg/s",
            val=0.72,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":air_flow_rate"
        ] = np.clip(
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_consumption_max"
            ],
            0.6,
            np.inf,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]

        clipped_flow_rate = np.clip(
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_consumption_max"
            ],
            0.6,
            np.inf,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + primary_heat_exchanger_id
            + ":air_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
        ] = np.where(
            clipped_flow_rate
            == inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_consumption_max"
            ],
            1.0,
            1e-6,
        )
