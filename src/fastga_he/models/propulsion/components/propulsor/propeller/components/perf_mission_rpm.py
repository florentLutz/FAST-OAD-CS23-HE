# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
    POINTS_NB_DESCENT,
)


class PerformancesRPMMission(om.ExplicitComponent):
    """
    Component which takes the desired rpm for propeller operation from the data and gives it the
    right format for the mission. It was deemed best to put it this way rather than the original
    way to simplify the construction of the power train file.

    The input rpm can either be a float (then during the whole mission the rpm is going to be the
    same), an array of three element (different rpm for the whole climb, whole cruise and whole
    descent) or an array of number of points elements for the individual control of each point.
    """

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_mission",
            val=np.nan,
            units="min**-1",
            desc="RPM of the propeller for the points",
            shape_by_conn=True,
        )

        self.add_output("rpm", units="min**-1", val=2500.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        rpm_mission = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_mission"
        ]

        if len(rpm_mission) == 1:
            outputs["rpm"] = np.full(number_of_points, rpm_mission)

        elif len(rpm_mission) == 3:
            outputs["rpm"] = np.concatenate(
                (
                    np.full(POINTS_NB_CLIMB, rpm_mission[0]),
                    np.full(POINTS_NB_CRUISE, rpm_mission[1]),
                    np.full(POINTS_NB_DESCENT, rpm_mission[2]),
                )
            )

        elif len(rpm_mission) == number_of_points:
            outputs["rpm"] = rpm_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:propeller:"
                + propeller_id
                + ":rpm_mission"
                + " should be 1, 3 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        rpm_mission = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_mission"
        ]

        if len(rpm_mission) == 1:
            partials[
                "rpm", "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_mission"
            ] = np.full(number_of_points, 1.0)

        elif len(rpm_mission) == 3:
            tmp_partials = np.zeros((number_of_points, 3))
            tmp_partials[:POINTS_NB_CLIMB, 0] = 1
            tmp_partials[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE, 1] = 1
            tmp_partials[POINTS_NB_CLIMB + POINTS_NB_CRUISE :, 2] = 1
            partials[
                "rpm", "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_mission"
            ] = tmp_partials

        elif len(rpm_mission) == number_of_points:
            partials[
                "rpm", "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_mission"
            ] = np.eye(number_of_points)
