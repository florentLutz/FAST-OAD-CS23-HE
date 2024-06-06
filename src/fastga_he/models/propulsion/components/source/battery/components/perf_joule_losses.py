# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesCellJouleLosses(om.ExplicitComponent):
    """
    Computation of the Joules losses inside the battery. It represents irreversible losses inside
    the battery and is always exothermic, see :cite:`cicconi:2017`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("internal_resistance", units="ohm", val=np.full(number_of_points, np.nan))
        self.add_input("current_one_module", units="A", val=np.full(number_of_points, np.nan))

        self.add_output("joule_losses_cell", units="W", val=np.full(number_of_points, 1))

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["joule_losses_cell"] = (
            inputs["internal_resistance"] * inputs["current_one_module"] ** 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["joule_losses_cell", "internal_resistance"] = inputs["current_one_module"] ** 2.0
        partials["joule_losses_cell", "current_one_module"] = (
            2.0 * inputs["internal_resistance"] * inputs["current_one_module"]
        )
