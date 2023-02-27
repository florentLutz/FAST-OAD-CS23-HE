# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDCSSPCEfficiency(om.ExplicitComponent):
    """
    This module computes the efficiency of the SSPC.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "closed",
            default=True,
            desc="Boolean to choose whether the breaker is closed or not.",
            types=bool,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "power_losses",
            val=np.full(number_of_points, np.nan),
            units="W",
        )
        self.add_input(
            "power_flow",
            val=np.full(number_of_points, np.nan),
            units="W",
            desc="Power at the terminal of the SSPC with the greatest voltage",
        )

        self.add_output(
            "efficiency",
            val=np.full(number_of_points, 1.0),
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]

        if self.options["closed"]:
            efficiency = np.where(
                inputs["power_flow"] > 1.0, 1.0 - inputs["power_losses"] / inputs["power_flow"], 1.0
            )
        else:
            efficiency = np.ones(number_of_points)

        outputs["efficiency"] = efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        if self.options["closed"]:
            partials["efficiency", "power_losses"] = -np.diag(1.0 / inputs["power_flow"])
            partials["efficiency", "power_flow"] = np.diag(
                inputs["power_losses"] / inputs["power_flow"] ** 2.0
            )
        else:
            partials["efficiency", "power_losses"] = np.zeros((number_of_points, number_of_points))
            partials["efficiency", "power_flow"] = np.zeros((number_of_points, number_of_points))
