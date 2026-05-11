# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .fluid_characteristics import FluidSpecificHeatCapacity


class PerformancesCompressedAirHeat(om.Group):
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

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

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
            "compressed_air_heat",
            _CompressedAirHeat(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
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

        self.add_output("air_inlet_temperature", val=370.0, units="K")
        self.add_output("air_outlet_temperature", val=345.0, units="K")
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


class _CompressedAirHeat(om.ExplicitComponent):
    """
    Compute the heat from air compression.
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
            units="J/kg/K",
            val=np.nan,
        )
        self.add_input(
            "air_inlet_temperature",
            units="K",
            val=np.nan,
        )
        self.add_input(
            "air_outlet_temperature",
            units="K",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":max_compressed_air_heat",
            units="W",
            val=0.72,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        inlet_temperature = inputs["air_inlet_temperature"]
        outlet_temperature = inputs["air_outlet_temperature"]
        mean_specific_heat_capacity = inputs["air_mean_specific_heat_capacity"]
        flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":max_compressed_air_heat"
        ] = flow_rate * mean_specific_heat_capacity * (inlet_temperature - outlet_temperature)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        inlet_temperature = inputs["air_inlet_temperature"]
        outlet_temperature = inputs["air_outlet_temperature"]
        mean_specific_heat_capacity = inputs["air_mean_specific_heat_capacity"]
        flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":max_compressed_air_heat",
            "air_inlet_temperature",
        ] = flow_rate * mean_specific_heat_capacity

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":max_compressed_air_heat",
            "air_outlet_temperature",
        ] = -flow_rate * mean_specific_heat_capacity

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":max_compressed_air_heat",
            "air_mean_specific_heat_capacity",
        ] = flow_rate * (inlet_temperature - outlet_temperature)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":max_compressed_air_heat",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
        ] = mean_specific_heat_capacity * (inlet_temperature - outlet_temperature)
