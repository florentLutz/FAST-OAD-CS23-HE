#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCABatteryDistanceToTargetCapacityLoss(om.ImplicitComponent):
    """
    Class that computes the distance to the target relative capacity loss at the end of the life of
    the battery and updates the number of cycles
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
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":end_of_life_relative_capacity_loss",
            val=0.4,
            units="unitless",
            desc="Relative capacity loss criteria for deciding of battery end of life",
        )
        self.add_input(
            name="capacity_loss_total",
            val=np.nan,
            units="unitless",
            desc="Capacity lost due to total aging",
        )

        self.add_output(
            name="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            val=500.0,
            units="unitless",
            desc="Expected lifetime of the battery pack, expressed in cycles. Default value is the "
            "number of cycle required for the reference cell to reach 60% nominal capacity",
        )

    def setup_partials(self):
        battery_pack_id = self.options["battery_pack_id"]
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            wrt="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":end_of_life_relative_capacity_loss",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            wrt="capacity_loss_total",
            val=-1.0,
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        battery_pack_id = self.options["battery_pack_id"]

        residuals[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"
        ] = (
            inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":end_of_life_relative_capacity_loss"
            ]
            - inputs["capacity_loss_total"]
        )
