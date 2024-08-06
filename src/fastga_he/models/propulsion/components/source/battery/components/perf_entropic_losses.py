# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesCellEntropicLosses(om.ExplicitComponent):
    """
    Computation of the entropic losses inside the battery. It represents reversible losses inside
    the battery and can be either endothermic or exothermic, depending on the entropic heat
    coefficient and the sign of the current going in or our of the battery, see
    :cite:`cicconi:2017`. The sign convention used here is that the battery discharging means a
    positive current so the losses will be of the opposite sign of the entropic heat coefficient,
    see :cite:`damay:2016`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "entropic_heat_coefficient", units="V/degK", val=np.full(number_of_points, np.nan)
        )
        self.add_input("current_one_module", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("cell_temperature", units="degK", val=np.full(number_of_points, np.nan))

        self.add_output("entropic_losses_cell", units="W", val=np.full(number_of_points, 1))

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["entropic_losses_cell"] = (
            -inputs["current_one_module"]
            * inputs["cell_temperature"]
            * inputs["entropic_heat_coefficient"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["entropic_losses_cell", "current_one_module"] = -(
            inputs["cell_temperature"] * inputs["entropic_heat_coefficient"]
        )
        partials["entropic_losses_cell", "cell_temperature"] = -(
            inputs["current_one_module"] * inputs["entropic_heat_coefficient"]
        )
        partials["entropic_losses_cell", "entropic_heat_coefficient"] = -(
            inputs["current_one_module"] * inputs["cell_temperature"]
        )
