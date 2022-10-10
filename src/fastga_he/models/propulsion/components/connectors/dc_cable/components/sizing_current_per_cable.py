# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingCurrentPerCable(om.ExplicitComponent):
    """Computation of max current per cable ."""

    def initialize(self):

        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
            val=1.0,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            val=np.nan,
            units="A",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:current",
            units="A",
        )

        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:current"
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max"]
            / inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:current",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
        ] = (
            1.0
            / inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
        )

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:current",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
        ] = -(
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max"]
            / inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
            ** 2.0
        )
