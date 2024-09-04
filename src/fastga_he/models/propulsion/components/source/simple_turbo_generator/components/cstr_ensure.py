# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_TURBO_GENERATOR_POWER,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_TURBO_GENERATOR_POWER,
    "fastga_he.submodel.propulsion.constraints.turbo_generator.power.ensure",
)
class ConstraintsPowerEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum power seen by the generator during the
    mission and the value used for sizing, ensuring each component works below its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

    def setup(self):
        turbo_generator_id = self.options["turbo_generator_id"]

        self.add_input(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max",
            units="kW",
            val=np.nan,
            desc="Maximum value of the shaft power the turbo generator receives",
        )
        self.add_input(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
            units="kW",
            val=np.nan,
            desc="Max continuous shaft power of the turbo generator",
        )
        self.add_output(
            "constraints:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
            units="kW",
            val=0.0,
            desc="Respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
            wrt=[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":shaft_power_max",
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":power_rating",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]

        outputs[
            "constraints:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating"
        ] = (
            inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":shaft_power_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":power_rating"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]

        partials[
            "constraints:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
        ] = -1.0
