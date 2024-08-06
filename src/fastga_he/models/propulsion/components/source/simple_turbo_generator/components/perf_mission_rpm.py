# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import scipy as sp
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesRPMMission(om.ExplicitComponent):
    """
    Component which takes the desired rpm for generator operation from the data and gives it the
    right format for the mission. It was deemed best to put it this way rather than the original
    way to simplify the construction of the power train file.

    The input rpm can either be a float (then during the whole mission the rpm is going to be the
    same) or an array of number of points elements for the individual control of each point.
    """

    def initialize(self):
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        turbo_generator_id = self.options["turbo_generator_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":rpm_mission",
            val=np.nan,
            units="min**-1",
            desc="RPM of the propeller for the points",
            shape_by_conn=True,
        )

        self.add_output("rpm", units="min**-1", val=2500.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]
        number_of_points = self.options["number_of_points"]

        rpm_mission = inputs[
            "data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":rpm_mission"
        ]

        if len(rpm_mission) == 1:
            outputs["rpm"] = np.full(number_of_points, rpm_mission)

        elif len(rpm_mission) == number_of_points:
            outputs["rpm"] = rpm_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":rpm_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]
        number_of_points = self.options["number_of_points"]

        rpm_mission = inputs[
            "data:propulsion:he_power_train:turbo_generator:" + turbo_generator_id + ":rpm_mission"
        ]

        if len(rpm_mission) == 1:
            partials[
                "rpm",
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":rpm_mission",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "rpm",
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":rpm_mission",
            ] = sp.sparse.eye(number_of_points, format="csc")
