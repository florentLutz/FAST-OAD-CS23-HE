# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesBatteryEfficiency(om.ExplicitComponent):
    """
    Computation of efficiency of the battery based on the losses at battery level and the output
    voltage and current.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("losses_battery", units="W", val=np.full(number_of_points, np.nan))
        self.add_input("power_out", units="W", val=np.full(number_of_points, np.nan))

        self.add_output("efficiency", val=np.full(number_of_points, 1.0), lower=0.0, upper=1.0)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Doing it like this instead of using where straight up allows to avoid computing the
        # division by zero, should there be any. With where, in addition to the condition, both
        # array are fully computed
        efficiency = np.ones_like(inputs["power_out"])
        np.divide(
            inputs["power_out"] - inputs["losses_battery"],
            inputs["power_out"],
            out=efficiency,
            where=np.abs(inputs["power_out"]) >= 200.0,
        )

        outputs["efficiency"] = efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials_losses = np.full_like(inputs["power_out"], 1e-6)
        np.divide(
            -1.0,
            inputs["power_out"],
            out=partials_losses,
            where=np.abs(inputs["power_out"]) >= 200.0,
        )
        partials["efficiency", "losses_battery"] = np.diag(partials_losses)

        partials_current_out = np.full_like(inputs["power_out"], 1e-6)
        np.divide(
            inputs["losses_battery"],
            inputs["power_out"] ** 2.0,
            out=partials_current_out,
            where=np.abs(inputs["power_out"]) >= 200.0,
        )
        partials["efficiency", "power_out"] = np.diag(partials_current_out)
