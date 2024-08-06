# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBatteryVolume(om.ExplicitComponent):
    """
    Computation of the volume the battery based on the volume of the modules.
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
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:volume",
            units="L",
            val=np.nan,
            desc="Volume of one module of the battery",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            val=np.nan,
            desc="Number of modules in parallel inside the battery pack",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            units="L",
            val=400.0,
            desc="Volume of the battery pack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume"] = (
            inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
            ]
            * inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:volume"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
        ] = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:volume"
        ]
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:volume",
        ] = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
        ]
