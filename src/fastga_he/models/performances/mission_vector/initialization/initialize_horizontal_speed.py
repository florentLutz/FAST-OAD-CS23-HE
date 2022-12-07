# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class InitializeHorizontalSpeed(om.ExplicitComponent):
    """Initializes the horizontal airspeed at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "true_airspeed",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="m/s",
        )
        self.add_input(
            "gamma", shape=number_of_points, val=np.full(number_of_points, 0.0), units="deg"
        )

        self.add_output("horizontal_speed", val=np.full(number_of_points, 50.0), units="m/s")

    def setup_partials(self):

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        true_airspeed = inputs["true_airspeed"]
        gamma = inputs["gamma"] * np.pi / 180.0

        outputs["horizontal_speed"] = true_airspeed * np.cos(gamma)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        true_airspeed = inputs["true_airspeed"]
        gamma = inputs["gamma"] * np.pi / 180.0

        partials["horizontal_speed", "gamma"] = (
            -np.diag(true_airspeed * np.sin(gamma)) * np.pi / 180.0
        )
        partials["horizontal_speed", "true_airspeed"] = np.diag(np.cos(gamma))
