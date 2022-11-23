# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterHeatSinkTubeMassFlow(om.ExplicitComponent):
    """
    Computation of the maximum mass flow that will be required in the heat sink tube, should be a
    result of the performance group but as the thermal management will not be detailed for now,
    it is going to be computed here based on the maximum heat to dissipate. Method from
    :cite:`giraud:2014`
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:specific_heat_capacity",
            units="J/degK/kg",
            val=3260.0,
            desc="Specific heat capacity of the coolant fluid",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:density",
            units="kg/m**3",
            val=1082.0,
            desc="Density of the coolant fluid",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
            units="W",
            val=np.nan,
            desc="Maximum power losses of the inverter (all modules)",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:temperature_in_rating",
            units="degK",
            val=np.nan,
            desc="Density of the coolant fluid",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:temperature_out_rating",
            units="degK",
            val=np.nan,
            desc="Density of the coolant fluid",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:max_mass_flow",
            units="m**3/s",
            val=0.1,
            desc="Maximum mass flow necessary to cool the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        inverter_losses_max = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat"
        ]
        specific_heat_capacity_coolant = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:specific_heat_capacity"
        ]
        density_coolant = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:coolant:density"
        ]
        delta_t_coolant = (
            inputs[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink:coolant:temperature_out_rating"
            ]
            - inputs[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink:coolant:temperature_in_rating"
            ]
        )

        outputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:max_mass_flow"
        ] = inverter_losses_max / (
            delta_t_coolant * specific_heat_capacity_coolant * density_coolant
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        inverter_losses_max = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat"
        ]
        specific_heat_capacity_coolant = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:specific_heat_capacity"
        ]
        density_coolant = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:coolant:density"
        ]
        delta_t_coolant = (
            inputs[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink:coolant:temperature_out_rating"
            ]
            - inputs[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink:coolant:temperature_in_rating"
            ]
        )

        partials[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:max_mass_flow",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:specific_heat_capacity",
        ] = -inverter_losses_max / (
            delta_t_coolant * specific_heat_capacity_coolant ** 2.0 * density_coolant
        )
        partials[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:max_mass_flow",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:coolant:density",
        ] = -inverter_losses_max / (
            delta_t_coolant * specific_heat_capacity_coolant * density_coolant ** 2.0
        )
        partials[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:max_mass_flow",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:temperature_out_rating",
        ] = -inverter_losses_max / (
            delta_t_coolant ** 2.0 * specific_heat_capacity_coolant * density_coolant
        )
        partials[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:max_mass_flow",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:temperature_in_rating",
        ] = inverter_losses_max / (
            delta_t_coolant ** 2.0 * specific_heat_capacity_coolant * density_coolant
        )
        partials[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:coolant:max_mass_flow",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
        ] = 1.0 / (delta_t_coolant * specific_heat_capacity_coolant * density_coolant)
