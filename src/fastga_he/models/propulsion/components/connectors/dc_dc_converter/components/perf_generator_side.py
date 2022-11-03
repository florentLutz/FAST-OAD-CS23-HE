# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesConverterGeneratorSide(om.ImplicitComponent):
    """
    The converter is divided between a load side where the battery is and a generator side where
    the rest of the circuit is. This component represents the generator side.

    Based on the methodology from :cite:`hendricks:2019`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "voltage_target",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Target voltage at the output side of the converter, solely used for convergence",
        )
        self.add_input(
            "voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="voltage to output side",
        )

        self.add_output(
            "current_out",
            val=np.full(number_of_points, 400.0),
            units="A",
            desc="current to output side",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(self, inputs, outputs, residuals):

        residuals["current_out"] = inputs["voltage_target"] - inputs["voltage_out"]

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        partials["current_out", "voltage_target"] = np.eye(number_of_points)
        partials["current_out", "voltage_out"] = -np.eye(number_of_points)
