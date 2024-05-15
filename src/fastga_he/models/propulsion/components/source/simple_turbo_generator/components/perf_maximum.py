# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum shaft power of the turbo generator.
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

        turbo_generator_id = self.options["turbo_generator_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("shaft_power_in", units="W", val=np.nan, shape=number_of_points)

        self.add_output(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max",
            units="W",
            val=42000.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max",
            wrt="shaft_power_in",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turbo_generator_id = self.options["turbo_generator_id"]

        outputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max"
        ] = np.max(inputs["shaft_power_in"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        turbo_generator_id = self.options["turbo_generator_id"]

        partials[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max",
            "shaft_power_in",
        ] = np.where(inputs["shaft_power_in"] == np.max(inputs["shaft_power_in"]), 1.0, 0.0)
