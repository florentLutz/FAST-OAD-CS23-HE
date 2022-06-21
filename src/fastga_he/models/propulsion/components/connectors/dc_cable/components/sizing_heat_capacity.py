# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from .sizing_harness_mass import LENGTH_FACTOR


class HeatCapacityCable(om.ExplicitComponent):
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
            + ":cable:heat_capacity_per_length",
            val=np.nan,
            units="J/degK/m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            val=np.nan,
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity",
            val=1,
            units="J/degK",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        heat_capacity = (
            inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:heat_capacity_per_length"
            ]
            * inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * LENGTH_FACTOR
        )

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:heat_capacity"
        ] = heat_capacity

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity_per_length",
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * LENGTH_FACTOR
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:heat_capacity_per_length"
            ]
            * LENGTH_FACTOR
        )
