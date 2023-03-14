# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingGeneratorDiameter(om.ExplicitComponent):
    """Computation of the diameter of a the generator."""

    def initialize(self):
        # Reference generator : EMRAX 268

        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )
        self.options.declare(
            "diameter_ref",
            default=0.268,
            desc="Diameter of the reference generator in [m]",
        )

    def setup(self):

        generator_id = self.options["generator_id"]

        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:diameter",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":diameter",
            val=self.options["diameter_ref"],
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":diameter",
            wrt="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:diameter",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        generator_id = self.options["generator_id"]
        d_ref = self.options["diameter_ref"]

        d_scaling = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:diameter"
        ]

        outputs["data:propulsion:he_power_train:generator:" + generator_id + ":diameter"] = (
            d_ref * d_scaling
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        generator_id = self.options["generator_id"]
        d_ref = self.options["diameter_ref"]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":diameter",
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:diameter",
        ] = d_ref
