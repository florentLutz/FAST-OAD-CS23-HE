# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from .sizing_harness_mass import LENGTH_FACTOR


class SizingReferenceResistance(om.ExplicitComponent):
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
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:resistance_per_length",
            val=np.nan,
            units="ohm/m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            val=np.nan,
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:resistance",
            val=1e-3,
            units="ohm",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        resistance = (
            inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:resistance_per_length"
            ]
            * inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * LENGTH_FACTOR
        )

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:resistance"
        ] = resistance

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:resistance",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:resistance_per_length",
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * LENGTH_FACTOR
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:resistance",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:resistance_per_length"
            ]
            * LENGTH_FACTOR
        )
