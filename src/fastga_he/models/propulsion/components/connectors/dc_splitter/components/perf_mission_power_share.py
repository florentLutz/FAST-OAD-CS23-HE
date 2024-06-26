# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import scipy as sp
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesMissionPowerShare(om.ExplicitComponent):
    """
    Component which takes the desired power share for the splitter operation from the data and
    gives it the right format for the mission. It was deemed best to put it this way rather than
    the original way to simplify the construction of the power train file.

    The input power share can either be a float (then during the whole mission the power share is
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
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_share",
            val=np.nan,
            units="W",
            desc="Share of the power going to the first (primary) input, in W",
            shape_by_conn=True,
        )

        self.add_output(
            "power_share",
            units="W",
            val=150e3,
            shape=number_of_points,
            desc="Share of the power going to the first (primary) input, in W, with a format "
            "adapted to mission. If below nothing will go in the secondary input, if above, "
            "the complement will flow in the secondary input",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]
        number_of_points = self.options["number_of_points"]

        power_split = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_share"
        ]

        if len(power_split) == 1:
            outputs["power_share"] = np.full(number_of_points, power_split)

        elif len(power_split) == number_of_points:
            outputs["power_share"] = power_split

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":power_share"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]
        number_of_points = self.options["number_of_points"]

        power_split = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_share"
        ]

        if len(power_split) == 1:
            partials[
                "power_share",
                "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_share",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "power_share",
                "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":power_share",
            ] = sp.sparse.eye(number_of_points)
