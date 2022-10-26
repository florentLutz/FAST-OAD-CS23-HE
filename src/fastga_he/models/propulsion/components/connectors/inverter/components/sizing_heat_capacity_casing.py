# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterCasingsWeight(om.ExplicitComponent):
    """
    Computation of the heat capacity of one casing. Very simplistic approach based on a degraded
    specific heat capacity of Si materials (material of the IGBT) to account for the presence of
    other materials in the casing. Material properties are taken from :cite:`tan:2022`.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:mass",
            units="kg",
            val=np.nan,
            desc="Weight of the casings (3 of them in the inverter)",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":casing:specific_heat",
            units="J/kg/degK",
            val=600.0,
            desc="Equivalent specific heat capacity of the casing",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:heat_capacity",
            units="J/degK",
            val=200.0,
            desc="Heat capacity of one casing",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:heat_capacity"
        ] = (
            inputs[
                "settings:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":casing:specific_heat"
            ]
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:mass"]
        ) / 3.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:heat_capacity",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:mass",
        ] = (
            inputs[
                "settings:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":casing:specific_heat"
            ]
            / 3.0
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:heat_capacity",
            "settings:propulsion:he_power_train:inverter:" + inverter_id + ":casing:specific_heat",
        ] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:mass"] / 3.0
        )
