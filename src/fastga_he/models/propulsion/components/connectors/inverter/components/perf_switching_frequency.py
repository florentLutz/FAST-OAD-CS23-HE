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


class PerformancesSwitchingFrequencyMission(om.ExplicitComponent):
    """
    Component which takes the desired switching frequency for inverter operation from the data
    and gives it the right format for the mission. It was deemed best to put it this way rather
    than the original way to simplify the construction of the power train file.

    The input switching frequency can either be a float (then during the whole mission the
    frequency is going to be the same), an array of three element (different frequency for the
    whole climb, whole cruise and whole descent) or an array of number of points elements for the
    individual control of each point.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":switching_frequency_mission",
            val=np.nan,
            units="Hz",
            desc="Switching frequency of the inverter for the points",
            shape_by_conn=True,
        )

        self.add_output("switching_frequency", units="Hz", val=15.0e3, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        f_switch_mission = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":switching_frequency_mission"
        ]

        if len(f_switch_mission) == 1:
            outputs["switching_frequency"] = np.full(number_of_points, f_switch_mission)

        elif len(f_switch_mission) == 3:
            outputs["switching_frequency"] = np.concatenate(
                (
                    np.full(POINTS_NB_CLIMB, f_switch_mission[0]),
                    np.full(POINTS_NB_CRUISE, f_switch_mission[1]),
                    np.full(POINTS_NB_DESCENT, f_switch_mission[2]),
                )
            )

        elif len(f_switch_mission) == number_of_points:
            outputs["switching_frequency"] = f_switch_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":switching_frequency_mission"
                + " should be 1, 3 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        f_switch_mission = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":switching_frequency_mission"
        ]

        if len(f_switch_mission) == 1:
            partials[
                "switching_frequency",
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":switching_frequency_mission",
            ] = np.full(number_of_points, 1.0)

        elif len(f_switch_mission) == 3:
            tmp_partials = np.zeros((number_of_points, 3))
            tmp_partials[:POINTS_NB_CLIMB, 0] = 1
            tmp_partials[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE, 1] = 1
            tmp_partials[POINTS_NB_CLIMB + POINTS_NB_CRUISE :, 2] = 1
            partials[
                "switching_frequency",
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":switching_frequency_mission",
            ] = tmp_partials

        elif len(f_switch_mission) == number_of_points:
            partials[
                "switching_frequency",
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":switching_frequency_mission",
            ] = np.eye(number_of_points)
