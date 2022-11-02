# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesElectricalNode(om.ImplicitComponent):
    """
    Simple DC bus based on Kirchoff's current law, ensure that all the current going into the bus
    goes out of it by modulating its voltage. Current sign convention is that input current going
    in is positive and output current going out is positive.

    Based on :cite:`hendricks:2019`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="number_of_inputs",
            default=1,
            types=int,
            desc="Number of connections at the input of the bus",
            allow_none=False,
        )
        self.options.declare(
            name="number_of_outputs",
            default=1,
            types=int,
            desc="Number of connections at the output of the bus",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        for i in range(self.options["number_of_inputs"]):

            # Choice was made to start current numbering at 1 to irritate any future programmer
            # working on the code
            self.add_input(
                name="current_in_" + str(i + 1),
                units="A",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="Current going into the bus at input number " + str(i + 1),
            )

        for j in range(self.options["number_of_outputs"]):

            self.add_input(
                name="current_out_" + str(j + 1),
                units="A",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="Current going out of the bus at output number " + str(j + 1),
            )

        self.add_output(
            name="voltage", val=np.ones(number_of_points), units="V", desc="Voltage of the bus"
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(self, inputs, outputs, residuals):

        number_of_points = self.options["number_of_points"]

        # Kirchoff's law: the sum of current is zero with the sign convention we defined

        residuals["voltage"] = np.zeros(number_of_points)

        for i in range(self.options["number_of_inputs"]):
            residuals["voltage"] += inputs["current_in_" + str(i + 1)]

        for j in range(self.options["number_of_outputs"]):
            residuals["voltage"] -= inputs["current_out_" + str(j + 1)]

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        for i in range(self.options["number_of_inputs"]):
            partials["voltage", "current_in_" + str(i + 1)] = np.eye(number_of_points)

        for j in range(self.options["number_of_outputs"]):
            partials["voltage", "current_out_" + str(j + 1)] = -np.eye(number_of_points)
