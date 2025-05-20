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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":power_ac_out_max",
            units="kW",
            val=np.nan,
            desc="Max power at the output side of the inverter",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost",
            units="USD",
            val=3500.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        power_rating = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":power_ac_out_max"
        ]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost"] = (
            4666.0 * power_rating**0.0928
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":power_ac_out_max",
        ] = (
            -433.0048
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":power_ac_out_max"]
            ** -0.9072
        )
