# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHeatSinkTubeMassFlow(om.ExplicitComponent):
    """
    Computation of the maximum mass flow that will be required in the heat sink tube, should be a
    result of the performance group but as the thermal management will not be detailed for now,
    it is going to be computed here based on the maximum heat to dissipate. Method from
    :cite:`giraud:2014`
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use a heatsink",
            allow_none=False,
        )

    def setup(self):

        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":heat_sink:coolant:specific_heat_capacity",
            units="J/degK/kg",
            val=3260.0,
            desc="Specific heat capacity of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:density",
            units="kg/m**3",
            val=1082.0,
            desc="Density of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":dissipable_heat",
            units="W",
            val=np.nan,
            desc="Maximum power losses of the inverter (all modules)",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:temperature_in_rating",
            units="degK",
            val=np.nan,
            desc="Density of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:temperature_out_rating",
            units="degK",
            val=np.nan,
            desc="Density of the coolant fluid",
        )

        self.add_output(
            name=prefix + ":heat_sink:coolant:max_mass_flow",
            units="m**3/s",
            val=0.1,
            desc="Maximum mass flow necessary to cool the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        inverter_losses_max = inputs[prefix + ":dissipable_heat"]
        specific_heat_capacity_coolant = inputs[
            prefix + ":heat_sink:coolant:specific_heat_capacity"
        ]
        density_coolant = inputs[prefix + ":heat_sink:coolant:density"]
        delta_t_coolant = (
            inputs[prefix + ":heat_sink:coolant:temperature_out_rating"]
            - inputs[prefix + ":heat_sink:coolant:temperature_in_rating"]
        )

        outputs[prefix + ":heat_sink:coolant:max_mass_flow"] = inverter_losses_max / (
            delta_t_coolant * specific_heat_capacity_coolant * density_coolant
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        prefix = self.options["prefix"]

        inverter_losses_max = inputs[prefix + ":dissipable_heat"]
        specific_heat_capacity_coolant = inputs[
            prefix + ":heat_sink:coolant:specific_heat_capacity"
        ]
        density_coolant = inputs[prefix + ":heat_sink:coolant:density"]
        delta_t_coolant = (
            inputs[prefix + ":heat_sink:coolant:temperature_out_rating"]
            - inputs[prefix + ":heat_sink:coolant:temperature_in_rating"]
        )

        partials[
            prefix + ":heat_sink:coolant:max_mass_flow",
            prefix + ":heat_sink:coolant:specific_heat_capacity",
        ] = -inverter_losses_max / (
            delta_t_coolant * specific_heat_capacity_coolant ** 2.0 * density_coolant
        )
        partials[
            prefix + ":heat_sink:coolant:max_mass_flow", prefix + ":heat_sink:coolant:density"
        ] = -inverter_losses_max / (
            delta_t_coolant * specific_heat_capacity_coolant * density_coolant ** 2.0
        )
        partials[
            prefix + ":heat_sink:coolant:max_mass_flow",
            prefix + ":heat_sink:coolant:temperature_out_rating",
        ] = -inverter_losses_max / (
            delta_t_coolant ** 2.0 * specific_heat_capacity_coolant * density_coolant
        )
        partials[
            prefix + ":heat_sink:coolant:max_mass_flow",
            prefix + ":heat_sink:coolant:temperature_in_rating",
        ] = inverter_losses_max / (
            delta_t_coolant ** 2.0 * specific_heat_capacity_coolant * density_coolant
        )
        partials[prefix + ":heat_sink:coolant:max_mass_flow", prefix + ":dissipable_heat"] = 1.0 / (
            delta_t_coolant * specific_heat_capacity_coolant * density_coolant
        )
