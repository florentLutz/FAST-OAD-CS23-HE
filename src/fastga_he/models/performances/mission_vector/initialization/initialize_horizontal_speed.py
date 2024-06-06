# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class InitializeHorizontalSpeed(om.ExplicitComponent):
    """Initializes the horizontal airspeed at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points_climb", default=1, desc="number of equilibrium to be treated in climb"
        )
        self.options.declare(
            "number_of_points_cruise",
            default=1,
            desc="number of equilibrium to be treated in " "cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=1,
            desc="number of equilibrium to be treated in descent",
        )
        self.options.declare(
            "number_of_points_reserve",
            default=1,
            desc="number of equilibrium to be treated in reserve",
        )

    def setup(self):
        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

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

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        true_airspeed = inputs["true_airspeed"]
        gamma = inputs["gamma"] * np.pi / 180.0

        outputs["horizontal_speed"] = true_airspeed * np.cos(gamma)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        true_airspeed = inputs["true_airspeed"]
        gamma = inputs["gamma"] * np.pi / 180.0

        partials["horizontal_speed", "gamma"] = -(true_airspeed * np.sin(gamma)) * np.pi / 180.0
        partials["horizontal_speed", "true_airspeed"] = np.cos(gamma)
