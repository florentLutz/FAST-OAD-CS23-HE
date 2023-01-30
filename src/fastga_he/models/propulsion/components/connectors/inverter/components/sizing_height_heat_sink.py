# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterHeatSinkHeight(om.ExplicitComponent):
    """
    Computation of the height of the heat sink of the inverter (plaque froide in french).
    Implementation of the formula from :cite:`giraud:2014`.
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
            + ":heat_sink:tube:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the tube for the cooling of the inverter",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:height",
            units="m",
            val=0.20,
            desc="Height of the heat sink",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:height"] = (
            1.5
            * inputs[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink:tube:outer_diameter"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:height",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:outer_diameter",
        ] = 1.5
