# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesCryogenicHydrogenTankConduction(om.ExplicitComponent):
    """
    Computation of the amount of the amount of hydrogen boil-off during the mission.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "heat_convection",
            units="W",
            val=np.full(number_of_points, np.nan),
            desc="Tank exterior heat convection at each time step",
        )

        self.add_input(
            "heat_radiation",
            units="W",
            val=np.full(number_of_points, np.nan),
            desc="Tank exterior heat radiation at each time step",
        )

        self.add_output(
            "heat_conduction",
            units="W",
            val=np.full(number_of_points, 17.23),
            desc="Tank wall heat conduction at each time step",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["heat_conduction"] = inputs["heat_convection"] + inputs["heat_radiation"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]

        partials["heat_conduction", "heat_convection"] = np.ones(number_of_points)

        partials["heat_conduction", "heat_radiation"] = np.ones(number_of_points)
