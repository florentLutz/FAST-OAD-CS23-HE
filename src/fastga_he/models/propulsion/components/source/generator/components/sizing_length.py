# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingGeneratorLength(om.ExplicitComponent):
    """Computation of the length of a cylindrical generator."""

    def initialize(self):
        # Reference generator : EMRAX 268

        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )
        self.options.declare(
            "length_ref", default=0.091, desc="Length of the reference generator in [m]"
        )

    def setup(self):
        generator_id = self.options["generator_id"]

        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:length",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":length",
            val=self.options["length_ref"],
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":length",
            wrt="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:length",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]

        l_scaling = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:length"
        ]

        l_ref = self.options["length_ref"]

        outputs["data:propulsion:he_power_train:generator:" + generator_id + ":length"] = (
            l_ref * l_scaling
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        generator_id = self.options["generator_id"]

        l_ref = self.options["length_ref"]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":length",
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:length",
        ] = l_ref
