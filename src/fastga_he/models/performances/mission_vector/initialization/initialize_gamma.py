# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import logging

import numpy as np
import openmdao.api as om

_LOGGER = logging.getLogger(__name__)


class InitializeGamma(om.ExplicitComponent):
    """Initializes the climb angle at each time step."""

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

        self.add_input("vertical_speed", val=np.full(number_of_points, np.nan), units="m/s")
        self.add_input(
            "true_airspeed",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="m/s",
        )

        self.add_output("gamma", val=np.full(number_of_points, 0.0), units="rad")

        self.declare_partials(
            of="gamma",
            wrt=["vertical_speed", "true_airspeed"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        true_airspeed = inputs["true_airspeed"]
        vertical_speed = inputs["vertical_speed"]

        gamma = np.arcsin(vertical_speed / true_airspeed)

        outputs["gamma"] = gamma

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        true_airspeed = inputs["true_airspeed"]
        vertical_speed = inputs["vertical_speed"]

        partials["gamma", "true_airspeed"] = (
            -1.0
            / np.sqrt(1.0 - np.square(vertical_speed / true_airspeed))
            * vertical_speed
            / true_airspeed**2.0
        )
        partials["gamma", "vertical_speed"] = (
            1.0 / np.sqrt(1.0 - np.square(vertical_speed / true_airspeed)) / true_airspeed
        )
