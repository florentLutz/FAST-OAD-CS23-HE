# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_TURBO_GENERATOR_POWER,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_TURBO_GENERATOR_POWER] = (
    "fastga_he.submodel.propulsion.constraints.turbo_generator.power.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_TURBO_GENERATOR_POWER,
    "fastga_he.submodel.propulsion.constraints.turbo_generator.power.enforce",
)
class ConstraintsPowerEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum power seen by the generator during the mission is used for
    the sizing, ensuring a fitted design for the power of each component.
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
            desc="Maximum value of the power the turbo generator receives",
        )

        self.add_output(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
            units="kW",
            val=250.0,
            desc="Max continuous power of the turbo generator",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":power_rating",
            wrt="data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]

        outputs[
            "data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":power_rating"
        ] = inputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max"
        ]
