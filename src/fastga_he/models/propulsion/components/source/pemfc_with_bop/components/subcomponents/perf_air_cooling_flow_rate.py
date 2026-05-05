# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .fluid_characteristics.fluid_specific_heat_capacity import FluidSpecificHeatCapacity


class PerformancesAirCoolingFlowRate(om.Group):
    """
    Compute the air cooling flow rate.
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
            name="connected_inlet_id",
            default=None,
            desc="Identifier of the inlet connected to the air cooling flow rate",
            allow_none=False,
        )
        self.options.declare(
            name="connected_heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger connected to the air cooling flow rate",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        coolant_fluid_type = self.options["coolant_fluid_type"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connect_inlet_id = self.options["connected_inlet_id"]
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]

        self.add_subsystem(
            "performances_mean_temperature",
            _MeanTemperature(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "performances_mean_pressure",
            _MeanPressure(number_of_points=number_of_points, pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "mean_air_specific_heat_capacity",
            FluidSpecificHeatCapacity(fluid="air"),
            promotes=[
                ("fluid_temperature", "mean_air_temperature"),
                ("fluid_pressure", "mean_air_pressure"),
                ("fluid_specific_heat_capacity", "mean_air_specific_heat_capacity"),
            ],
        )
        self.add_subsystem(
            "mean_coolant_specific_heat_capacity",
            FluidSpecificHeatCapacity(fluid=coolant_fluid_type),
            promotes=[
                ("fluid_temperature", "mean_coolant_temperature"),
                ("fluid_pressure", "mean_coolant_pressure"),
                ("fluid_specific_heat_capacity", "mean_coolant_specific_heat_capacity"),
            ],
        )
        self.add_subsystem(
            "performances_air_cooling_flow_rate",
            _PerformancesAirCoolingFlowRate(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "performances_air_mass_flow_factor",
            PerformancesAirMassFlowFactor(
                number_of_points=number_of_points,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                connected_inlet_id=connect_inlet_id,
            ),
            promotes=["*"],
        )


class _MeanTemperature(om.ExplicitComponent):
    """
    Computes the mean temperature of the coolant side and the air side.
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
            + ":coolant:outlet_temperature",
            units="K",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            units="K",
            val=np.nan,
        )

        self.add_output(
            name="mean_coolant_temperature",
            units="K",
            val=335.0,
        )
        self.add_output(
            name="mean_air_temperature",
            units="K",
            val=315.0,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            of="mean_coolant_temperature",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature",
            val=0.5,
        )

        self.declare_partials(
            of="mean_coolant_temperature",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            val=0.5,
        )

        self.declare_partials(
            of="mean_air_temperature",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            val=0.5,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs["mean_coolant_temperature"] = 0.5 * (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":coolant:outlet_temperature"
            ]
            + inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":coolant:intermediate_temperature"
            ]
        )
        outputs["mean_air_temperature"] = 0.5 * (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":coolant:intermediate_temperature"
            ]
            + 295.0
        )


class _MeanPressure(om.ExplicitComponent):
    """
    Computes the mean pressure of the coolant side and the air side.
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

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:static_pressure",
            units="Pa",
            val=np.nan,
        )
        self.add_input("ambient_pressure", units="Pa", val=np.full(number_of_points, np.nan))

        self.add_output(
            name="mean_coolant_pressure",
            units="Pa",
            val=200000.0,
        )
        self.add_output(
            name="mean_air_pressure",
            units="Pa",
            val=101325.0,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            of="mean_coolant_pressure",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:static_pressure",
            val=1.0,
        )

        self.declare_partials(
            of="mean_air_pressure",
            wrt="ambient_pressure",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs["mean_coolant_pressure"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:static_pressure"
        ]
        outputs["mean_air_pressure"] = np.max(inputs["ambient_pressure"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["mean_air_pressure", "ambient_pressure"] = np.where(
            inputs["ambient_pressure"] == np.max(inputs["ambient_pressure"]), 1.0, 1e-6
        )


class _PerformancesAirCoolingFlowRate(om.ExplicitComponent):
    """
    Computes the air cooling flow rate.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="connected_heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger connected to the air cooling flow rate",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature",
            units="K",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            units="K",
            val=np.nan,
        )
        self.add_input("mean_air_specific_heat_capacity", units="J/kg/K", val=np.nan)
        self.add_input("mean_coolant_specific_heat_capacity", units="J/kg/K", val=np.nan)
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            units="kg/s",
            val=np.nan,
            desc="Mass flow rate of the PEMFC coolant during the mission",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":heat_exchanger_efficiency",
            val=0.98,
            units="unitless",
        )

        self.add_output(
            name="air_cooling_flow_rate",
            units="kg/s",
            val=0.1,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]

        efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":heat_exchanger_efficiency"
        ]
        coolant_outlet_temperature = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature"
        ]
        coolant_intermediate_temperature = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature"
        ]
        mean_air_specific_heat_capacity = inputs["mean_air_specific_heat_capacity"]
        mean_coolant_specific_heat_capacity = inputs["mean_coolant_specific_heat_capacity"]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]

        outputs["air_cooling_flow_rate"] = (
            efficiency
            * mean_coolant_specific_heat_capacity
            * coolant_mass_flow_rate
            * (coolant_outlet_temperature - 300.0)
            / (mean_air_specific_heat_capacity * (coolant_intermediate_temperature - 1.0 - 300.0))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]

        efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":heat_exchanger_efficiency"
        ]
        coolant_outlet_temperature = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature"
        ]
        coolant_intermediate_temperature = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature"
        ]
        mean_air_specific_heat_capacity = inputs["mean_air_specific_heat_capacity"]
        mean_coolant_specific_heat_capacity = inputs["mean_coolant_specific_heat_capacity"]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]

        common_denominator = mean_air_specific_heat_capacity * (
            coolant_intermediate_temperature - 1.0 - 300.0
        )

        partials[
            "air_cooling_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":heat_exchanger_efficiency",
        ] = (
            mean_coolant_specific_heat_capacity
            * coolant_mass_flow_rate
            * (coolant_outlet_temperature - 300.0)
            / common_denominator
        )
        partials[
            "air_cooling_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature",
        ] = (
            efficiency
            * mean_coolant_specific_heat_capacity
            * coolant_mass_flow_rate
            / common_denominator
        )

        partials[
            "air_cooling_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
        ] = (
            efficiency
            * mean_coolant_specific_heat_capacity
            * coolant_mass_flow_rate
            * (300.0 - coolant_outlet_temperature)
            * mean_air_specific_heat_capacity
            / common_denominator**2.0
        )

        partials["air_cooling_flow_rate", "mean_coolant_specific_heat_capacity"] = (
            efficiency
            * coolant_mass_flow_rate
            * (coolant_outlet_temperature - 300.0)
            / common_denominator
        )

        partials[
            "air_cooling_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
        ] = (
            efficiency
            * mean_coolant_specific_heat_capacity
            * (coolant_outlet_temperature - 300.0)
            / common_denominator
        )

        partials["air_cooling_flow_rate", "mean_air_specific_heat_capacity"] = -(
            efficiency
            * mean_coolant_specific_heat_capacity
            * coolant_mass_flow_rate
            * (coolant_outlet_temperature - 300.0)
            * (coolant_intermediate_temperature - 1.0 - 300.0)
            / common_denominator**2.0
        )


class PerformancesAirMassFlowFactor(om.ExplicitComponent):
    """
    Computes the air mass flow factor, which is the ratio of the air cooling flow rate to the coolant mass flow rate.
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
            name="connected_inlet_id",
            default=None,
            desc="Identifier of the inlet connected to the air cooling flow rate",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["connected_inlet_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="air_cooling_flow_rate",
            units="kg/s",
            val=np.nan,
        )
        self.add_input(
            "air_consumption",
            units="kg/s",
            shape=number_of_points,
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass_flow_factor",
            val=3.0,
            units="unitless",
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="air_consumption",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            method="exact",
        )
        self.declare_partials(
            of="*",
            wrt="air_cooling_flow_rate",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["connected_inlet_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass_flow_factor"
        ] = np.max(inputs["air_cooling_flow_rate"] / inputs["air_consumption"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["connected_inlet_id"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass_flow_factor",
            "air_cooling_flow_rate",
        ] = 1.0 / np.min(inputs["air_consumption"])

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass_flow_factor",
            "air_consumption",
        ] = np.where(
            np.min(inputs["air_consumption"]) == inputs["air_consumption"],
            -inputs["air_cooling_flow_rate"] / inputs["air_consumption"] ** 2.0,
            1e-6,
        )
