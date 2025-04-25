# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCInverterCost(om.ExplicitComponent):
    """
    Computation of the inverter purchase cost. Based on the retail price provided by
    https://www.mcico.com/truebluepower/inverters?purchase_type=New+Outright%2CNew+Exchange.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":power_rating_max",
            units="kW",
            val=np.nan,
            desc="Power rating of the inverter",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":cost_per_unit",
            units="USD",
            val=3500.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        power_rating = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":power_rating_max"
        ]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":cost_per_unit"] = (
            2167.0 * np.log(power_rating) + 6910.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":cost_per_unit",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":power_rating_max",
        ] = (
            2167.0
            / inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":power_rating_max"]
        )
