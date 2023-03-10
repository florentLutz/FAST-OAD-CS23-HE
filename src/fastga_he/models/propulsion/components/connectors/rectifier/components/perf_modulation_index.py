# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesModulationIndex(om.ExplicitComponent):
    """
    Component which computes the value of the modulation index of the rectifier.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_voltage_peak_in",
            units="V",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Peak line to neutral voltage at the input of the rectifier",
        )
        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the output side of the rectifier",
        )

        self.add_output(
            "modulation_index",
            val=np.full(number_of_points, 1.0),
            desc="Modulation index of the rectifier",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["modulation_index"] = inputs["dc_voltage_out"] / inputs["ac_voltage_peak_in"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["modulation_index", "dc_voltage_out"] = np.diag(1.0 / inputs["ac_voltage_peak_in"])
        partials["modulation_index", "ac_voltage_peak_in"] = -np.diag(
            inputs["dc_voltage_out"] / inputs["ac_voltage_peak_in"] ** 2.0
        )
