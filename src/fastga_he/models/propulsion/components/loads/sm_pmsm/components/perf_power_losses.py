# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPowerLosses(om.ExplicitComponent):
    """
    Computation of the total motor power losses as sum of the mechanical, iron, and joule
    losses.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "joule_power_losses",
            units="kW",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "iron_power_losses",
            units="kW",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "mechanical_power_losses",
            units="kW",
            val=np.nan,
            shape=number_of_points,
        )

        self.add_output(
            "power_losses",
            units="kW",
            val=0.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="power_losses",
            wrt=[
                "joule_power_losses",
                "iron_power_losses",
                "mechanical_power_losses",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["power_losses"] = (
            inputs["mechanical_power_losses"]
            + inputs["iron_power_losses"]
            + inputs["joule_power_losses"]
        )
