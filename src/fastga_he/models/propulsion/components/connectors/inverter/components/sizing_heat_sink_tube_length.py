# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterHeatSinkTubeLength(om.ExplicitComponent):
    """
    Computation of the heat sink tube useful length in the inverter. Implementation of the
    formula from :cite:`giraud:2014`.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:length",
            units="m",
            val=np.nan,
            desc="Length of the heat sink",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":tube:number_of_passes",
            val=4,
            desc="Number of passes in the heat sink (between 2 and 6 usually)",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:length",
            units="m",
            val=0.20,
            desc="Length of the tube which is useful for the cooling of the inverter",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:length",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:length",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":tube:number_of_passes",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:length"
        ] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:length"]
            * inputs[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":tube:number_of_passes"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:length",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:length",
        ] = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":tube:number_of_passes"
        ]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:length",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":tube:number_of_passes",
        ] = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:length"]
