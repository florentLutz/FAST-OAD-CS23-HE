# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class LCCBatteryPackOperationalCost(om.ExplicitComponent):
    """
    Computation of the battery pack annual operational cost.
    """

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            name="data:TLAR:flight_per_year",
            val=np.nan,
            desc="Average number of flight per year",
        )
        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            val=500.0,
            desc="Expected lifetime of the battery pack, expressed in cycles. Default value is the "
            "number of cycle required for the reference cell to reach 60% nominal capacity",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost",
            units="USD",
            val=np.nan,
            desc="battery pack cost per unit",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":operational_cost",
            units="USD/yr",
            val=500.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":operational_cost"
        ] = (
            inputs["data:TLAR:flight_per_year"]
            * inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost"
            ]
            / inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":operational_cost",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost",
        ] = (
            inputs["data:TLAR:flight_per_year"]
            / inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"]
        )

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":operational_cost",
            "data:TLAR:flight_per_year",
        ] = (
            inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost"
            ]
            / inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"]
        )

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":operational_cost",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
        ] = (
            -inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost"
            ]
            * inputs["data:TLAR:flight_per_year"]
            / inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"]
            ** 2.0
        )
