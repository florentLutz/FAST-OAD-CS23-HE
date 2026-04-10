# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .subcomponents.fluid_characteristics import FluidSpecificHeatCapacity


class PerformancesPEMFCStackBOPCoolantMassFlowRate(om.Group):
    """
    Coolant temperature computation of the PEMFC stack, using for BOP and TMS sizing.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "coolant_temperature_gradiant",
            default=10.0,
            desc="The temperature difference of the PEMFC coolant I/O [K]",
        )
        self.options.declare(
            "coolant_fluid_type",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )

    def setup(self):
        coolant_temperature_gradiant = self.options["coolant_temperature_gradiant"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_type = self.options["coolant_fluid_type"]

        self.add_subsystem(
            "coolant_specific_heat_capacity",
            FluidSpecificHeatCapacity(fluid=coolant_type),
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
            "coolant_mass_flow_rate",
            _CoolantMassFlowRate(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                coolant_temperature_gradiant=coolant_temperature_gradiant,
            ),
            promotes=["*"],
        )


class _CoolantMassFlowRate(om.ExplicitComponent):
    """
    Coolant mass flow rate computation of the TMS.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "coolant_temperature_gradiant",
            default=10.0,
            desc="The temperature difference of the PEMFC coolant I/O [K]",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power of the PEMFC stack has to provide during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max",
            units="kW",
            val=np.nan,
            desc="Maximum total thermal power of the PEMFC stack has to provide during the mission",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_rating",
            units="kW",
            val=0.0,
        )
        self.add_input(
            "mean_coolant_specific_heat_capacity",
            units="kJ/kg/K",
            val=np.nan,
            desc="Mean coolant specific heat capacity of the PEMFC coolant during the mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            units="kg/s",
            val=4.1,
            desc="Mass flow rate of the PEMFC coolant during the mission",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_temperature_gradiant = self.options["coolant_temperature_gradiant"]

        power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max"
        ]
        thermal_power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max"
        ]
        bop_power = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_rating"
        ]

        unclipped_heat_dissipation = thermal_power_max - power_max - bop_power
        clipped_heat_dissipation = np.clip(unclipped_heat_dissipation, 0.0, 240.0)
        mean_coolant_specific_heat_capacity = inputs["mean_coolant_specific_heat_capacity"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ] = clipped_heat_dissipation / (
            mean_coolant_specific_heat_capacity * coolant_temperature_gradiant
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_temperature_gradiant = self.options["coolant_temperature_gradiant"]

        power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max"
        ]
        thermal_power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max"
        ]
        bop_power = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_rating"
        ]
        mean_coolant_specific_heat_capacity = inputs["mean_coolant_specific_heat_capacity"]

        unclipped_heat_dissipation = thermal_power_max - power_max - bop_power
        clipped_heat_dissipation = np.clip(unclipped_heat_dissipation, 0.0, 240.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max",
        ] = np.where(
            unclipped_heat_dissipation == clipped_heat_dissipation,
            -1.0 / (mean_coolant_specific_heat_capacity * coolant_temperature_gradiant),
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_rating",
        ] = np.where(
            unclipped_heat_dissipation == clipped_heat_dissipation,
            -1.0 / (mean_coolant_specific_heat_capacity * coolant_temperature_gradiant),
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max",
        ] = np.where(
            unclipped_heat_dissipation == clipped_heat_dissipation,
            1.0 / (mean_coolant_specific_heat_capacity * coolant_temperature_gradiant),
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            "mean_coolant_specific_heat_capacity",
        ] = -clipped_heat_dissipation / (
            mean_coolant_specific_heat_capacity**2.0 * coolant_temperature_gradiant
        )
