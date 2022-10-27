# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

LENGTH_FACTOR = 1.2  # For the mass and resistance computation we will consider that the cable is
# longer than it actually is to account for the fact that he is stranded to shield EMI


class SizingHarnessMass(om.ExplicitComponent):
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
            + ":cable:mass_per_length",
            val=np.nan,
            units="kg/m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
            val=1,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":contactor:mass",
            units="kg",
            val=np.nan,
            desc="Mass of all the contactors in the harness",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":mass",
            val=1,
            units="kg",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        mass = (
            inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:mass_per_length"
            ]
            * inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
            * LENGTH_FACTOR
            + inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":contactor:mass"
            ]
        )

        outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":mass"] = mass

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":mass",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:mass_per_length",
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
            * LENGTH_FACTOR
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":mass",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:mass_per_length"
            ]
            * LENGTH_FACTOR
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":mass",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:mass_per_length"
            ]
            * LENGTH_FACTOR
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":mass",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":contactor:mass",
        ] = 1.0
