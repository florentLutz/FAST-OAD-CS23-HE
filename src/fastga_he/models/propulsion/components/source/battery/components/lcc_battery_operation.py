# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class LCCBatteryPackOperation(om.ExplicitComponent):
    """
    Computation of the battery pack operation cost and electricity cost. The charging cost is
    estimated from https://eniplenitude.eu/e-mobility/pricing.
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
            name="data:cost:operation:mission_per_year",
            val=np.nan,
            units="1/yr",
            desc="Flight mission per year",
        )

        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_mission",
            units="kW*h",
            val=np.nan,
            desc="Energy drawn from the battery for the mission",
        )

        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            val=500.0,
            desc="Expected lifetime of the inverter, expressed in cycles. Default value is the "
            "number of cycle required for the reference cell to reach 60% nominal capacity",
        )

        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cost_per_unit",
            units="USD",
            val=np.nan,
            desc="battery pack cost per unit",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":maintenance_per_unit",
            units="USD/yr",
            val=500.0,
            desc="battery pack cost per unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":maintenance_per_unit"
        ] = (
            inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cost_per_unit"
            ]
            / inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"]
            + 0.655
            * inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":energy_consumed_mission"
            ]
        ) * inputs["data:cost:operation:mission_per_year"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        energy_consumed = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_mission"
        ]

        cost = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cost_per_unit"
        ]

        lifespan = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"
        ]

        mission_per_year = inputs["data:cost:operation:mission_per_year"]

        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":maintenance_per_unit",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cost_per_unit",
        ] = mission_per_year / lifespan

        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":maintenance_per_unit",
            "data:cost:operation:mission_per_year",
        ] = cost / lifespan + 0.655 * energy_consumed

        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":maintenance_per_unit",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
        ] = -cost * mission_per_year / lifespan**2.0

        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":maintenance_per_unit",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_mission",
        ] = 0.655 * mission_per_year
