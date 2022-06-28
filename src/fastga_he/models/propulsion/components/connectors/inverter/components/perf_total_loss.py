# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesLosses(om.ExplicitComponent):
    """Computation of Conduction losses for the IGBT and the diode."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "conduction_losses_diode",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )
        self.add_input(
            "conduction_losses_IGBT",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )
        self.add_input(
            "switching_losses_diode",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )
        self.add_input(
            "switching_losses_IGBT",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )

        self.add_output(
            "losses_inverter",
            units="W",
            val=np.full(number_of_points, 0.0),
            shape=number_of_points,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["losses_inverter"] = 6 * (
            inputs["switching_losses_IGBT"]
            + inputs["switching_losses_diode"]
            + inputs["conduction_losses_IGBT"]
            + inputs["conduction_losses_diode"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["losses_inverter", "switching_losses_IGBT"] = 6.0 * np.eye(number_of_points)
        partials["losses_inverter", "switching_losses_diode"] = 6.0 * np.eye(number_of_points)
        partials["losses_inverter", "conduction_losses_IGBT"] = 6.0 * np.eye(number_of_points)
        partials["losses_inverter", "conduction_losses_diode"] = 6.0 * np.eye(number_of_points)
