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
            name="supplement_heat_exchanger_id",
            default=None,
            desc="Identifier of the supplement heat exchanger",
            allow_none=False,
        )
        self.options.declare(
            name="connected_air_inlet_id",
            default=None,
            desc="Identifier of the connected air flush_inlet",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_fluid_type = self.options["coolant_fluid_type"]
        supplement_heat_exchanger_id = self.options["supplement_heat_exchanger_id"]
        connected_air_inlet_id = self.options["connected_air_inlet_id"]

        self.add_subsystem(
            "supplement_heat_exchanger_air_properties",
            _SupplementHeatExchangerAirProperties(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "mean_air_temperature_hex_performances",
            _MeanAirTemperature(pemfc_stack_bop_id=pemfc_stack_bop_id),
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
            "design_air_mass_flow_hex_performances",
            _DesignAirMassFlow(
                pemfc_stack_bop_id=pemfc_stack_bop_id, connected_air_inlet_id=connected_air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "minimum_heat_capacity",
            _MinimumHeatCapacity(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_outlet_temperature_hex_performances",
            _AirOutletTemperate(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                supplement_heat_exchanger_id=supplement_heat_exchanger_id,
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
                supplement_heat_exchanger_id=supplement_heat_exchanger_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "tms_air_outlet_temperature_hex_performances",
            _TMSAirOutletTemperate(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )


class _SupplementHeatExchangerAirProperties(om.ExplicitComponent):
    """
    Compute the specific heat capacity of the air at the supplement heat exchanger flush_inlet and outlet.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "diffuser_exit_temperature",
            val=np.nan,
            units="K",
            shape=number_of_points,
        )
        self.add_input(
            "diffuser_exit_pressure",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )

        self.add_output("air_inlet_temperature", val=300.0, units="K")
        self.add_output("air_static_pressure", val=101325.0, units="Pa")

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="air_static_pressure",
            wrt="diffuser_exit_pressure",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="air_inlet_temperature",
            wrt="diffuser_exit_temperature",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["air_static_pressure"] = np.max(inputs["diffuser_exit_pressure"])
        outputs["air_inlet_temperature"] = np.max(inputs["diffuser_exit_temperature"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        exit_temperature = inputs["diffuser_exit_temperature"]
        max_exit_temperature = np.max(exit_temperature)
        exit_pressure = inputs["diffuser_exit_pressure"]
        max_exit_pressure = np.max(exit_pressure)

        partials["air_inlet_temperature", "diffuser_exit_temperature"] = np.where(
            max_exit_temperature == exit_temperature, 1.0, 0.0
        )

        partials["air_static_pressure", "diffuser_exit_pressure"] = np.where(
            max_exit_pressure == exit_pressure, 1.0, 0.0
        )


class _MeanAirTemperature(om.ExplicitComponent):
    """
    Compute the mean temperature of the air at the supplement heat exchanger.
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

        self.add_input("air_inlet_temperature", val=np.nan, units="K")
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mean_temperature",
            val=np.nan,
            units="K",
        )

        self.add_output("mean_air_temperature", val=318.0, units="K")

    def setup_partials(self):
        self.declare_partials("*", "*", val=0.5)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs["mean_air_temperature"] = (
            inputs["air_inlet_temperature"]
            + inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":coolant:mean_temperature"
            ]
        ) / 2.0


class _DesignAirMassFlow(om.ExplicitComponent):
    """
    Compute the design air mass flow rate at the supplement heat exchanger flush_inlet.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="connected_air_inlet_id",
            default=None,
            desc="Identifier of the connected air flush_inlet",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_air_inlet_id = self.options["connected_air_inlet_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":design_air_mass_flow",
            val=np.nan,
            units="kg/s",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":mass_flow_factor",
            val=3.0,
            units="unitless",
        )

        self.add_output(
            "design_air_mass_flow",
            val=4.0,
            units="kg/s",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_air_inlet_id = self.options["connected_air_inlet_id"]

        inlet_design_air_mass_flow = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":design_air_mass_flow"
        ]
        mass_flow_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":mass_flow_factor"
        ]

        outputs["design_air_mass_flow"] = (
            inlet_design_air_mass_flow * mass_flow_factor / (mass_flow_factor + 1.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_air_inlet_id = self.options["connected_air_inlet_id"]

        inlet_design_air_mass_flow = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":design_air_mass_flow"
        ]
        mass_flow_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":mass_flow_factor"
        ]

        partials[
            "design_air_mass_flow",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":design_air_mass_flow",
        ] = mass_flow_factor / (mass_flow_factor + 1.0)

        partials[
            "design_air_mass_flow",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":mass_flow_factor",
        ] = inlet_design_air_mass_flow / (mass_flow_factor + 1.0) ** 2.0


class _MinimumHeatCapacity(om.ExplicitComponent):
    """
    Compute the minimum heat capacity of the supplement heat exchanger.
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
            "design_air_mass_flow",
            val=np.nan,
            units="kg/s",
        )
        self.add_input(
            "mean_air_specific_heat_capacity",
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
            val=3000.0,
            units="W/K",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        air_mass_flow_rate = inputs["design_air_mass_flow"]
        mean_air_specific_heat_capacity = inputs["mean_air_specific_heat_capacity"]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        mean_coolant_specific_heat_capacity = inputs["mean_coolant_specific_heat_capacity"]

        outputs["minimum_heat_capacity"] = min(
            air_mass_flow_rate * mean_air_specific_heat_capacity,
            coolant_mass_flow_rate * mean_coolant_specific_heat_capacity,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        air_mass_flow_rate = inputs["design_air_mass_flow"]
        mean_air_specific_heat_capacity = inputs["mean_air_specific_heat_capacity"]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        mean_coolant_specific_heat_capacity = inputs["mean_coolant_specific_heat_capacity"]
        min_heat_capacity = min(
            air_mass_flow_rate * mean_air_specific_heat_capacity,
            coolant_mass_flow_rate * mean_coolant_specific_heat_capacity,
        )

        if air_mass_flow_rate * mean_air_specific_heat_capacity == min_heat_capacity:
            partials["minimum_heat_capacity", "design_air_mass_flow"] = (
                mean_air_specific_heat_capacity
            )
            partials["minimum_heat_capacity", "mean_air_specific_heat_capacity"] = (
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
            partials["minimum_heat_capacity", "design_air_mass_flow"] = 0.0
            partials["minimum_heat_capacity", "mean_air_specific_heat_capacity"] = 0.0
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
    Compute the outlet temperature of the air at the supplement heat exchanger outlet.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="supplement_heat_exchanger_id",
            default=None,
            desc="Identifier of the supplement heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        supplement_heat_exchanger_id = self.options["supplement_heat_exchanger_id"]

        self.add_input(
            "air_inlet_temperature",
            val=np.nan,
            units="K",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature",
            val=np.nan,
            units="K",
        )
        self.add_input(
            "design_air_mass_flow",
            val=np.nan,
            units="kg/s",
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
            + supplement_heat_exchanger_id
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
            val=335.0,
            units="K",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        supplement_heat_exchanger_id = self.options["supplement_heat_exchanger_id"]

        air_inlet_temperature = inputs["air_inlet_temperature"]
        coolant_outlet_temperature = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature"
        ]
        design_air_mass_flow = inputs["design_air_mass_flow"]
        mean_air_specific_heat_capacity = inputs["mean_air_specific_heat_capacity"]
        mini_heat_capacity = inputs["minimum_heat_capacity"]
        heat_exchanger_efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + supplement_heat_exchanger_id
            + ":heat_exchanger_efficiency"
        ]

        outputs["air_outlet_temperature"] = (
            air_inlet_temperature
            + heat_exchanger_efficiency
            * mini_heat_capacity
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (design_air_mass_flow * mean_air_specific_heat_capacity)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        supplement_heat_exchanger_id = self.options["supplement_heat_exchanger_id"]

        air_inlet_temperature = inputs["air_inlet_temperature"]
        coolant_outlet_temperature = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature"
        ]
        design_air_mass_flow = inputs["design_air_mass_flow"]
        mean_air_specific_heat_capacity = inputs["mean_air_specific_heat_capacity"]
        mini_heat_capacity = inputs["minimum_heat_capacity"]
        heat_exchanger_efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + supplement_heat_exchanger_id
            + ":heat_exchanger_efficiency"
        ]

        partials["air_outlet_temperature", "air_inlet_temperature"] = (
            1.0
            - heat_exchanger_efficiency
            * mini_heat_capacity
            / (design_air_mass_flow * mean_air_specific_heat_capacity)
        )

        partials[
            "air_outlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature",
        ] = (
            heat_exchanger_efficiency
            * mini_heat_capacity
            / (design_air_mass_flow * mean_air_specific_heat_capacity)
        )

        partials[
            "air_outlet_temperature",
            "design_air_mass_flow",
        ] = (
            -heat_exchanger_efficiency
            * mini_heat_capacity
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (design_air_mass_flow**2.0 * mean_air_specific_heat_capacity)
        )

        partials["air_outlet_temperature", "mean_air_specific_heat_capacity"] = (
            -heat_exchanger_efficiency
            * mini_heat_capacity
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (design_air_mass_flow * mean_air_specific_heat_capacity**2.0)
        )

        partials["air_outlet_temperature", "minimum_heat_capacity"] = (
            heat_exchanger_efficiency
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (design_air_mass_flow * mean_air_specific_heat_capacity)
        )

        partials[
            "air_outlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + supplement_heat_exchanger_id
            + ":heat_exchanger_efficiency",
        ] = (
            mini_heat_capacity
            * (coolant_outlet_temperature - air_inlet_temperature)
            / (design_air_mass_flow * mean_air_specific_heat_capacity)
        )


class _TMSAirOutletTemperate(om.ExplicitComponent):
    """
    Compute the outlet temperature of the air at the supplement heat exchanger outlet for TMS.
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
            "air_outlet_temperature",
            val=np.nan,
            units="K",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_outlet_temperature",
            val=335.0,
            units="K",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_outlet_temperature"
        ] = inputs["air_outlet_temperature"]


class _CoolantTemperature(om.ExplicitComponent):
    """
    The computation of the coolant temperature at the supplement heat exchanger's flush_inlet and outlet.
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
            + ":coolant:outlet_temperature",
            val=np.nan,
            units="K",
        )

        self.add_output(
            name="coolant_inlet_temperature",
            val=349.6,
            units="K",
        )
        self.add_output(
            name="coolant_outlet_temperature",
            val=331.0,
            units="K",
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            "coolant_outlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature",
            val=1.0,
        )
        self.declare_partials(
            "coolant_inlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs["coolant_outlet_temperature"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:intermediate_temperature"
        ]
        outputs["coolant_inlet_temperature"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature"
        ]


class _AirFlowRate(om.ExplicitComponent):
    """
    Compute the air mass flow rate at the supplement heat exchanger flush_inlet.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="supplement_heat_exchanger_id",
            default=None,
            desc="Identifier of the supplement heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        supplement_heat_exchanger_id = self.options["supplement_heat_exchanger_id"]

        self.add_input(
            "design_air_mass_flow",
            val=np.nan,
            units="kg/s",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + supplement_heat_exchanger_id
            + ":air_flow_rate",
            units="kg/s",
            val=0.6,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        supplement_heat_exchanger_id = self.options["supplement_heat_exchanger_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + supplement_heat_exchanger_id
            + ":air_flow_rate"
        ] = inputs["design_air_mass_flow"]
