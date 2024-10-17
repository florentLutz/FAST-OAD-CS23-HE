# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCASpeedReducerProdWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="speed_reducer_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )

    def setup(self):
        speed_reducer_id = self.options["speed_reducer_id"]

        self.add_input(
            name="data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the speed reducer",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Mass of speed reducer required for a functional unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        speed_reducer_id = self.options["speed_reducer_id"]

        outputs[
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":mass_per_fu"
        ] = (
            inputs["data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        speed_reducer_id = self.options["speed_reducer_id"]

        partials[
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":mass_per_fu",
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"]
        partials[
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":mass"]
