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
        self.options.declare(
            "cut_off_voltage", default=2.6, desc="Cut-off voltage of the battery cells"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("open_circuit_voltage", units="V", val=np.full(number_of_points, np.nan))
        self.add_input("internal_resistance", units="ohm", val=np.full(number_of_points, np.nan))
        self.add_input("current_one_module", units="A", val=np.full(number_of_points, np.nan))

        self.add_output("terminal_voltage", units="V", val=np.full(number_of_points, 4.0))

        self.declare_partials(
            of="terminal_voltage",
            wrt=["internal_resistance", "current_one_module"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="terminal_voltage",
            wrt="open_circuit_voltage",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["terminal_voltage"] = np.clip(
            (
                inputs["open_circuit_voltage"]
                - inputs["internal_resistance"] * inputs["current_one_module"]
            ),
            np.full_like(inputs["open_circuit_voltage"], self.options["cut_off_voltage"]),
            np.full_like(inputs["open_circuit_voltage"], 5.0),
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["terminal_voltage", "internal_resistance"] = -inputs["current_one_module"]
        partials["terminal_voltage", "current_one_module"] = -inputs["internal_resistance"]
