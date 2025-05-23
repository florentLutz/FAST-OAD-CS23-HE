# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class LCCBatteryPackCost(om.ExplicitComponent):
    """
    Computation of the battery pack cost, considering cost reduction with production
    maturity. The cost reduction factor and estimated price are obtained from :cite:`Wesley:2023`.
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
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_mission",
            units="kW*h",
            val=np.nan,
            desc="Energy drawn from the battery for the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cost_median_2022",
            units="USD/kW/h",
            val=482.0,
            desc="The Cost median per kW*h of the Li-ion battery pack in 2022",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":years_since_2022",
            units="yr",
            val=3.0,
            desc="Number of years since 2022",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost",
            units="USD",
            val=50e3,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        cost_22 = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cost_median_2022"
        ]
        energy = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_mission"
        ]
        num_year = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":years_since_2022"
        ]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost"
        ] = 1.08 * num_year**-0.228 * cost_22 * energy

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        cost_22 = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cost_median_2022"
        ]
        energy = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_mission"
        ]
        num_year = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":years_since_2022"
        ]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":years_since_2022",
        ] = -0.24624 * num_year**-1.228 * cost_22 * energy

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cost_median_2022",
        ] = 1.08 * num_year**-0.228 * energy

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":purchase_cost",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_mission",
        ] = 1.08 * num_year**-0.228 * cost_22
