# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterWeight(om.ExplicitComponent):
    """
    Computation of the weight of the inverter as the sum of all its subcomponents. Method from
    :cite:`giraud:2014`.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:mass",
            units="kg",
            val=np.nan,
            desc="Mass of the heat sink, includes tubes and core",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:mass",
            val=np.nan,
            units="kg",
            desc="Mass of the capacitor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":control_card:mass",
            val=1.0,
            units="kg",
            desc="Weight of the control card, is generally constant, taken at 1 kg",
        )
        # self.add_input(
        #     name="data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:mass",
        #     units="kg",
        #     val=np.nan,
        #     desc="Mass of the 3 single phase inductor",
        # )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":contactor:mass",
            units="kg",
            val=np.nan,
            desc="Mass of the 3 contactors",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
            units="kg",
            val=40,
            desc="Mass of the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass"] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:mass"]
            + inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:mass"]
            + inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:mass"]
            + inputs[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":control_card:mass"
            ]
            # + inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:mass"]
            + inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":contactor:mass"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:mass",
        ] = 1.0
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:mass",
        ] = 1.0
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:mass",
        ] = 1.0
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":control_card:mass",
        ] = 1.0
        # partials[
        #     "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
        #     "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:mass",
        # ] = 1.0
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":contactor:mass",
        ] = 1.0
