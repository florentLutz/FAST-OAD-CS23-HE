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
        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("voltage_out", units="V", val=np.full(number_of_points, np.nan))

        self.add_output("efficiency", val=np.full(number_of_points, 1.0), lower=0.0, upper=1.0)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        efficiency = np.where(
            np.abs(inputs["dc_current_out"]) < 1.0,
            1.0,
            1.0 - inputs["losses_battery"] / (inputs["dc_current_out"] * inputs["voltage_out"]),
        )
        outputs["efficiency"] = efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials_losses = np.where(
            np.abs(inputs["dc_current_out"]) < 1.0,
            1e-6,
            -1.0 / (inputs["dc_current_out"] * inputs["voltage_out"]),
        )
        partials["efficiency", "losses_battery"] = np.diag(partials_losses)

        partials_current_out = np.where(
            np.abs(inputs["dc_current_out"]) < 1.0,
            1e-6,
            inputs["losses_battery"] / (inputs["dc_current_out"] ** 2.0 * inputs["voltage_out"]),
        )
        partials["efficiency", "dc_current_out"] = np.diag(partials_current_out)

        partials_voltage_out = np.where(
            np.abs(inputs["dc_current_out"]) < 1.0,
            1e-6,
            inputs["losses_battery"] / (inputs["dc_current_out"] * inputs["voltage_out"] ** 2.0),
        )
        partials["efficiency", "voltage_out"] = np.diag(partials_voltage_out)
