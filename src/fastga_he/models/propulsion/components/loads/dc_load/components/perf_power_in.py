# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import scipy as sp
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesPowerIn(om.ExplicitComponent):
    """
    Component which takes the desired power input from the data and gives it the right format for
    the mission. It was deemed best to put it this way rather than the original way to simplify
    the construction of the power train file.

    The input power can either be a float (then during the whole mission the power is going to be
    the same) or an array of number of points elements for the individual control of each point.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )

    def setup(self):

        aux_load_id = self.options["aux_load_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_in_mission",
            val=np.nan,
            units="kW",
            desc="Input power of the auxiliary loads",
            shape_by_conn=True,
        )

        self.add_output("power_in", units="kW", val=10.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        aux_load_id = self.options["aux_load_id"]
        number_of_points = self.options["number_of_points"]

        p_in_mission = inputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_in_mission"
        ]

        if len(p_in_mission) == 1:
            outputs["power_in"] = np.full(number_of_points, p_in_mission)

        elif len(p_in_mission) == number_of_points:
            outputs["power_in"] = p_in_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:aux_load:"
                + aux_load_id
                + ":power_in_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        aux_load_id = self.options["aux_load_id"]
        number_of_points = self.options["number_of_points"]

        p_in_mission = inputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_in_mission"
        ]

        if len(p_in_mission) == 1:
            partials[
                "power_in",
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_in_mission",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "power_in",
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_in_mission",
            ] = sp.sparse.eye(number_of_points, format="csc")
