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

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="waste_heat",
            val=np.nan,
            units="kW",
            desc="Heat from PEMFC to dissipate",
            shape=number_of_points,
        )

        self.add_output("power_in", units="kW", val=10.0, shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="waste_heat",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        clipped_waste_heat = np.clip(inputs["waste_heat"], 0.01, np.inf)
        outputs["power_in"] = 0.0767 * clipped_waste_heat**1.114

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        clipped_waste_heat = np.clip(inputs["waste_heat"], 0.01, np.inf)

        partials["power_in", "waste_heat"] = np.where(
            clipped_waste_heat == inputs["waste_heat"],
            0.0854438 * inputs["waste_heat"] ** 0.114,
            1e-6,
        )
