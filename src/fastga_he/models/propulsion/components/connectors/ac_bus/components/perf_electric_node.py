# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesElectricalNode(om.ImplicitComponent):
    """
    Simple AC bus based on Kirchoff's current law, ensure that all the current going into the bus
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
                name="ac_current_rms_in_one_phase_" + str(i + 1),
                units="A",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="RMS value of the current going into the bus at input number " + str(i + 1),
            )

        for j in range(self.options["number_of_outputs"]):

            self.add_input(
                name="ac_current_rms_out_one_phase_" + str(j + 1),
                units="A",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="RMS value of the current going through one phase of the bus at output number "
                + str(j + 1),
            )

        self.add_output(
            name="ac_voltage_rms",
            val=np.full(number_of_points, 350),
            units="V",
            desc="RMS value of the voltage of the bus",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(self, inputs, outputs, residuals):

        number_of_points = self.options["number_of_points"]

        # Kirchoff's law: the sum of current is zero with the sign convention we defined

        residuals["ac_voltage_rms"] = np.zeros(number_of_points)

        for i in range(self.options["number_of_inputs"]):
            residuals["ac_voltage_rms"] += inputs["ac_current_rms_in_one_phase_" + str(i + 1)]

        for j in range(self.options["number_of_outputs"]):
            residuals["ac_voltage_rms"] -= inputs["ac_current_rms_out_one_phase_" + str(j + 1)]

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        for i in range(self.options["number_of_inputs"]):
            partials["ac_voltage_rms", "ac_current_rms_in_one_phase_" + str(i + 1)] = np.eye(
                number_of_points
            )

        for j in range(self.options["number_of_outputs"]):
            partials["ac_voltage_rms", "ac_current_rms_out_one_phase_" + str(j + 1)] = -np.eye(
                number_of_points
            )
