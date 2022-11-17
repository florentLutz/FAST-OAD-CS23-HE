# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesCellLosses(om.ExplicitComponent):
    """
    Computation of the total losses inside one cell. It account for joules losses and entropic
    losses as the other one can be neglected for now see :cite:`cicconi:2017`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("joule_losses_cell", units="W", val=np.full(number_of_points, np.nan))
        self.add_input("entropic_losses_cell", units="W", val=np.full(number_of_points, np.nan))

        self.add_output("losses_cell", units="W", val=np.full(number_of_points, 1))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["losses_cell"] = inputs["joule_losses_cell"] + inputs["entropic_losses_cell"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["losses_cell", "joule_losses_cell"] = np.eye(number_of_points)
        partials["losses_cell", "entropic_losses_cell"] = np.eye(number_of_points)
