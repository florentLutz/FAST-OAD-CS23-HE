# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesConverterLoadSide(om.ExplicitComponent):
    """
    The converter is divided between a load side where the battery is and a generator side where
    the rest of the circuit is. This component represents the load side.

    Based on the methodology from :cite:`hendricks:2019`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        batman = np.full(number_of_points, np.nan)

        self.add_input(
            "dc_voltage_in",
            val=batman,
            units="V",
            desc="Voltage at the input side of the converter",
        )
        self.add_input(
            "power", val=batman, units="W", desc="Power at the input side of the converter"
        )

        self.add_output(
            "dc_current_in",
            val=np.full(number_of_points, 400.0),
            units="A",
            desc="Current at the input side of the converter",
            lower=1e-4,
        )

        self.declare_partials(of="dc_current_in", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["dc_current_in"] = inputs["power"] / inputs["dc_voltage_in"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["dc_current_in", "dc_voltage_in"] = -np.diag(
            inputs["power"] / inputs["dc_voltage_in"] ** 2
        )
        partials["dc_current_in", "power"] = np.diag(1.0 / inputs["dc_voltage_in"])
