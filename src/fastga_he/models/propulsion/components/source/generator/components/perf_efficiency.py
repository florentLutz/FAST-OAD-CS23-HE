# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesEfficiency(om.ExplicitComponent):
    """Computation of the efficiency from shaft power and power losses."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("shaft_power_in", units="W", val=np.nan, shape=number_of_points)
        self.add_input("power_losses", units="W", val=np.nan, shape=number_of_points)

        self.add_output(
            "efficiency",
            val=np.full(number_of_points, 1.0),
            shape=number_of_points,
            lower=0.75,
            upper=1.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # To avoid dividing by zero
        power_shaft_in = np.clip(inputs["shaft_power_in"], 1.0, None)

        # The model was based on an efficiency map where the minimum value was 0.86, we will clip
        # a little below just in case
        efficiency = np.clip(1.0 - inputs["power_losses"] / power_shaft_in, 0.75, 1.0)

        outputs["efficiency"] = efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # To avoid dividing by zero
        power_shaft_in = np.clip(inputs["shaft_power_in"], 1.0, None)

        efficiency_untouched = 1.0 - inputs["power_losses"] / power_shaft_in

        partials_shaft_power = np.where(
            np.clip(efficiency_untouched, 0.75, 1.0) != efficiency_untouched,
            1e-6,
            inputs["power_losses"] / power_shaft_in ** 2.0,
        )
        partials["efficiency", "shaft_power_in"] = np.diag(partials_shaft_power)

        partials_losses = np.where(
            np.clip(efficiency_untouched, 0.75, 1.0) != efficiency_untouched,
            1e-6,
            -1.0 / power_shaft_in,
        )
        partials["efficiency", "power_losses"] = np.diag(partials_losses)
