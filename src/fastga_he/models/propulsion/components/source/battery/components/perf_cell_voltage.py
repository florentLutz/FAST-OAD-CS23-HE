# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesCellVoltage(om.ExplicitComponent):
    """
    Computation of the terminal voltage of the battery cells inside one module. Assumes it can be
    computed as the open circuit voltage minus the internal resistance times the current. Common
    enough model, can be found for instance in :cite:`vratny:2013`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("open_circuit_voltage", units="V", val=np.full(number_of_points, np.nan))
        self.add_input("internal_resistance", units="ohm", val=np.full(number_of_points, np.nan))
        self.add_input("current_one_module", units="A", val=np.full(number_of_points, np.nan))

        self.add_output("terminal_voltage", units="V", val=np.full(number_of_points, 4.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["terminal_voltage"] = np.clip(
            (
                inputs["open_circuit_voltage"]
                - inputs["internal_resistance"] * inputs["current_one_module"]
            ),
            np.full_like(inputs["open_circuit_voltage"], 1.0),
            np.full_like(inputs["open_circuit_voltage"], 5.0),
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["terminal_voltage", "open_circuit_voltage"] = np.eye(number_of_points)
        partials["terminal_voltage", "internal_resistance"] = -np.diag(inputs["current_one_module"])
        partials["terminal_voltage", "current_one_module"] = -np.diag(inputs["internal_resistance"])
