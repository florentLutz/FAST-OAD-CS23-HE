# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesShaftPowerIn(om.ExplicitComponent):
    """
    Computation of the shaft power at the input of the generator, assumes fixed efficiency.
    Default value is taken from literature :cote:`pettes:2021`
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        turbo_generator_id = self.options["turbo_generator_id"]

        self.add_input(
            "active_power", units="W", val=np.full(number_of_points, np.nan), shape=number_of_points
        )
        self.add_input(
            "data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":efficiency",
            val=0.96,
        )

        self.add_output("shaft_power_in", units="W", val=500.0e3, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turbo_generator_id = self.options["turbo_generator_id"]

        outputs["shaft_power_in"] = (
            inputs["active_power"]
            / inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":efficiency"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]
        turbo_generator_id = self.options["turbo_generator_id"]

        partials["shaft_power_in", "active_power"] = (
            np.eye(number_of_points)
            / inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":efficiency"
            ]
        )
        partials[
            "shaft_power_in",
            "data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":efficiency",
        ] = -(
            inputs["active_power"]
            / inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":efficiency"
            ]
            ** 2.0
        )
