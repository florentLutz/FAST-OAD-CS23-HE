# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesLossesOneCable(om.ExplicitComponent):
    """Class to compute the losses in one cable, for now it will only be the Joule losses."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_current_one_cable",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="current of line",
            shape=number_of_points,
        )
        self.add_input(
            "resistance_per_cable",
            val=np.full(number_of_points, np.nan),
            units="ohm",
            desc="resistance of line",
        )

        self.add_output(
            "conduction_losses",
            val=np.full(number_of_points, 0.0),
            units="W",
            desc="Joule losses in one cable of the harness",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["conduction_losses"] = (
            inputs["resistance_per_cable"] * inputs["dc_current_one_cable"] ** 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["conduction_losses", "resistance_per_cable"] = np.diag(
            inputs["dc_current_one_cable"] ** 2.0
        )
        partials["conduction_losses", "dc_current_one_cable"] = np.diag(
            2.0 * inputs["resistance_per_cable"] * inputs["dc_current_one_cable"]
        )
