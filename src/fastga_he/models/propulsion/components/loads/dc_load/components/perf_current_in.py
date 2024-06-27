# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesCurrentIn(om.ExplicitComponent):
    """
    Simple use of the power formula. The power is an input and the current comes from the rest of
    the network.

    Based on the methodology from :cite:`hendricks:2019`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the input side of the load",
        )
        self.add_input(
            "power_in",
            val=np.full(number_of_points, np.nan),
            units="W",
            desc="Power at the input side of the load",
        )

        self.add_output(
            "dc_current_in",
            val=np.full(number_of_points, 20.0),
            units="A",
            desc="Current at the input side of the load",
            lower=-1000.0,
            upper=1000.0,
        )

        self.declare_partials(
            of="dc_current_in",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["dc_current_in"] = inputs["power_in"] / inputs["dc_voltage_in"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["dc_current_in", "dc_voltage_in"] = (
            -inputs["power_in"] / inputs["dc_voltage_in"] ** 2
        )
        partials["dc_current_in", "power_in"] = 1.0 / inputs["dc_voltage_in"]
