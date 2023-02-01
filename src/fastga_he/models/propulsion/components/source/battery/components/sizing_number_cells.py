# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBatteryNumberCells(om.ExplicitComponent):
    """
    Computation of the total number of cells inside the battery based on the number of cells in
    each module and number of modules.
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
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            val=np.nan,
            desc="Number of modules in parallel inside the battery pack",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
            val=np.nan,
            desc="Number of cells in series inside one battery module",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells",
            val=30000.0,
            desc="Total number of cells in the battery pack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells"
        ] = (
            inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ]
            * inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
        ] = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
        ]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
        ] = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells"
        ]
