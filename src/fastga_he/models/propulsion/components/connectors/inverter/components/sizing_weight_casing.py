# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterCasingsWeight(om.ExplicitComponent):
    """
    Computation of the weight of the casingS, which includes 3 IGBT modules plus their casing.
    Based on a  regression on the SEMIKRON family from :cite:`giraud:2014`.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of one arm of the inverter",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:mass",
            units="kg",
            val=0.350,
            desc="Weight of the casings (3 of them in the inverter)",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:mass"] = 3.0 * (
            0.175 + 4e-4 * current_caliber
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = 12e-4
