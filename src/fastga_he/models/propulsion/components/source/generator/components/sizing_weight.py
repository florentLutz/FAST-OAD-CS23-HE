# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingGeneratorWeight(om.ExplicitComponent):
    """
    Computation of the weight of a cylindrical generator. Based on a regression on the EMRAX family
    assuming there is a mechanical part which scale with respect to the torque
    :cite:`budinger:2012`, a magnetic part which scales with the torque ** 3/3.5
    :cite:`budinger:2012` and a constant part. Regression can be seen in
    ..methodology.torque_scaling.
    """

    def initialize(self):
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):
        # Reference generator : EMRAX 268

        generator_id = self.options["generator_id"]

        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":torque_rating",
            val=np.nan,
            units="N*m",
            desc="Max continuous torque of the generator",
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":mass",
            val=20.0,
            units="kg",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":mass",
            wrt="data:propulsion:he_power_train:generator:" + generator_id + ":torque_rating",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]

        torque_cont = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":torque_rating"
        ]

        mass = 2.8 + 9.54e-3 * torque_cont + 0.1632 * torque_cont ** (3.0 / 3.5)

        outputs["data:propulsion:he_power_train:generator:" + generator_id + ":mass"] = mass

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        generator_id = self.options["generator_id"]

        torque_cont = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":torque_rating"
        ]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":mass",
            "data:propulsion:he_power_train:generator:" + generator_id + ":torque_rating",
        ] = 9.54e-3 + 0.1632 * 3.0 / 3.5 * torque_cont ** (3.0 / 3.5 - 1.0)
