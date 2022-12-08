# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesHeatSinkTemperatureMission(om.ExplicitComponent):
    """
    Component which takes the desired heat sink temperature for inverter operation from the data
    and gives it the right format for the mission. It was deemed best to put it this way rather
    than the original way to simplify the construction of the power train file.

    The input heat sink temperature can either be a float (then during the whole mission the
    temperature is going to be the same) or an array of number of points elements for
    the individual control of each point.
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
            + ":heat_sink_temperature_mission",
            val=np.nan,
            units="degK",
            desc="Heat sink temperature of the inverter for the points",
            shape_by_conn=True,
        )

        self.add_output("heat_sink_temperature", units="degK", val=283.15, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        t_hs_mission = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink_temperature_mission"
        ]

        if len(t_hs_mission) == 1:
            outputs["heat_sink_temperature"] = np.full(number_of_points, t_hs_mission)

        elif len(t_hs_mission) == number_of_points:
            outputs["heat_sink_temperature"] = t_hs_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink_temperature_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        t_hs_mission = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink_temperature_mission"
        ]

        if len(t_hs_mission) == 1:
            partials[
                "heat_sink_temperature",
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink_temperature_mission",
            ] = np.full(number_of_points, 1.0)

        elif len(t_hs_mission) == number_of_points:
            partials[
                "heat_sink_temperature",
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink_temperature_mission",
            ] = np.eye(number_of_points)
