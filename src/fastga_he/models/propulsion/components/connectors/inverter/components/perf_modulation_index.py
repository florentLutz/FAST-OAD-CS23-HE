# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesModulationIndex(om.ImplicitComponent):
    """
    Computation of the modulation index, left as an implicit component to leave a degree of
    freedom to the code to converge.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "peak_ac_voltage",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Peak voltage on the AC side of the inverter",
        )
        self.add_input(
            "dc_voltage",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Bus voltage on the DC side of the inverter",
        )

        self.add_output("modulation_index", val=np.full(number_of_points, 0.7))

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(self, inputs, outputs, residuals):

        residuals["modulation_index"] = (
            inputs["peak_ac_voltage"] - outputs["modulation_index"] * inputs["dc_voltage"]
        )

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        partials["modulation_index", "peak_ac_voltage"] = np.eye(number_of_points)
        partials["modulation_index", "dc_voltage"] = -np.diag(outputs["modulation_index"])
        partials["modulation_index", "modulation_index"] = -np.diag(inputs["dc_voltage"])

    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        if any(
            np.clip(inputs["peak_ac_voltage"] / inputs["dc_voltage"], 0, 1.0)
            != inputs["peak_ac_voltage"] / inputs["dc_voltage"]
        ):

            outputs["modulation_index"] = np.clip(
                inputs["peak_ac_voltage"] / inputs["dc_voltage"], 0, 1.0
            )
