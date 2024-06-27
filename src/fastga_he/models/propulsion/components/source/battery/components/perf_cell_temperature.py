# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import scipy as sp
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesCellTemperatureMission(om.ExplicitComponent):
    """
    Component which takes the desired cell temperature for battery operation from the data and
    gives it the right format for the mission. It was deemed best to put it this way rather than
    the original way to simplify the construction of the power train file.

    The input cell temperature can either be a float (then during the whole mission the
    temperature is going to be the same) or an array of number of points elements for
    the individual control of each point.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell_temperature_mission",
            val=np.nan,
            units="degK",
            desc="Cell temperature of the battery for the points",
            shape_by_conn=True,
        )

        self.add_output("cell_temperature", units="degK", val=283.15, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_pack_id = self.options["battery_pack_id"]
        number_of_points = self.options["number_of_points"]

        t_cell_mission = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell_temperature_mission"
        ]

        if len(t_cell_mission) == 1:
            outputs["cell_temperature"] = np.full(number_of_points, t_cell_mission)

        elif len(t_cell_mission) == number_of_points:
            outputs["cell_temperature"] = t_cell_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell_temperature_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        battery_pack_id = self.options["battery_pack_id"]
        number_of_points = self.options["number_of_points"]

        t_cell_mission = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell_temperature_mission"
        ]

        if len(t_cell_mission) == 1:
            partials[
                "cell_temperature",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell_temperature_mission",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "cell_temperature",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell_temperature_mission",
            ] = sp.sparse.eye(number_of_points, format="csc")
