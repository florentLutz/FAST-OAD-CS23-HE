# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingTurboGeneratorWeight(om.ExplicitComponent):
    """
    Computation of the weight of a turbo generator. Default value based on specific density from
    :cite:`pettes:2021`.
    """

    def initialize(self):
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

    def setup(self):
        # Reference generator : EMRAX 268

        turbo_generator_id = self.options["turbo_generator_id"]

        self.add_input(
            name="data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
            val=np.nan,
            units="kW",
            desc="Max continuous power of the turbo generator",
        )
        self.add_input(
            name="data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_density",
            val=5.0,
            units="kW/kg",
            desc="Power density of the turbo generator",
        )

        self.add_output(
            name="data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":mass",
            val=20.0,
            units="kg",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":mass",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]

        outputs[
            "data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":mass"
        ] = (
            inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":power_rating"
            ]
            / inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":power_density"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]

        partials[
            "data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":mass",
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
        ] = (
            1.0
            / inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":power_density"
            ]
        )
        partials[
            "data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":mass",
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_density",
        ] = -(
            inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":power_rating"
            ]
            / inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":power_density"
            ]
            ** 2.0
        )
