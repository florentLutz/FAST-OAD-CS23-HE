# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import scipy as sp
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesMissionPowerSplit(om.ExplicitComponent):
    """
    Component which takes the desired power split for the splitter operation from the data and
    gives it the right format for the mission. It was deemed best to put it this way rather than
    the original way to simplify the construction of the power train file.

    The input power split can either be a float (then during the whole mission the power split is
    going to be the same) or an array of number of points elements for the individual control of
    each point.
    """

    def initialize(self):
        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        dc_splitter_id = self.options["dc_splitter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_split",
            val=np.nan,
            units="percent",
            desc="Percent of the power going to the first (primary) input, in %",
            shape_by_conn=True,
        )

        self.add_output(
            "power_split",
            units="percent",
            val=50.0,
            shape=number_of_points,
            desc="Percent of the power going to the first (primary) input, in %, with a format "
            "adapted to mission",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_splitter_id = self.options["dc_splitter_id"]
        number_of_points = self.options["number_of_points"]

        power_split = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_split"
        ]

        if len(power_split) == 1:
            outputs["power_split"] = np.full(number_of_points, power_split)

        elif len(power_split) == number_of_points:
            outputs["power_split"] = power_split

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":power_split"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_splitter_id = self.options["dc_splitter_id"]
        number_of_points = self.options["number_of_points"]

        power_split = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_split"
        ]

        if len(power_split) == 1:
            partials[
                "power_split",
                "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_split",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "power_split",
                "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_split",
            ] = sp.sparse.eye(number_of_points, format="csc")
