# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCInverterOperationalCost(om.ExplicitComponent):
    """
    Computation of the inverter annual operational cost.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost",
            units="USD",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":lifespan",
            units="yr",
            val=15.0,
            desc="Expected lifetime of the inverter, typically around 15 year",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost",
            units="USD/yr",
            val=350.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost"] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost"]
            / inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":lifespan"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        purchase_cost = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost"
        ]
        lifespan = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":lifespan"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost",
        ] = 1.0 / lifespan

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":lifespan",
        ] = -purchase_cost / lifespan**2.0
