# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDCSSPCVoltageOut(om.ExplicitComponent):
    """
    This step computes the voltage drop across the SSPC based on the current flowing through it
    and the resistance.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare("at_bus_output", default=1, desc="number of equilibrium to be treated")

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_current_in",
            val=np.full(number_of_points, np.nan),
            units="A",
        )
        self.add_input(
            "resistance_sspc",
            val=np.full(number_of_points, np.nan),
            units="ohm",
        )
        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
        )

        self.add_output(
            "dc_voltage_out",
            val=np.full(number_of_points, 500.0),
            units="V",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Let's talk about that option, shall we ?
        # Because the input/output of electrical components not always matching inputs/outputs at
        # the OpenMDAO format as part of the load flow analysis, there sometimes is very strange
        # connections to be done.
        # In our instance, the SSPC was thought implemented so as to isolate buses or
        # components connected to buses from the rest of the electric network. This means that
        # one side of the SSPC will match the output of a bus while the other will have to match
        # whatever component is connected. This means for instance that the same SSPC model
        # should work in this two places in the following example:
        # BUS 1 --- SSPC 1 --- HARNESS 1 --- SSPC 2 --- BUS 2
        # This means that we will have to choose a side of the SSPC that is always connected to
        # the bus, we will choose here the INPUT, but going back to that example, if power flows
        # from left to right and since current won't change (losses are translated in voltage
        # drop) the same formula should provide that V_in_sspc_1 < V_out_sspc_1 < V_out_sspc_2 <
        # V_in_sspc_2, which can't be the case unless we use a factor that change depending on
        # whether or not the SSPC is at the output of a bus (SSPC 2) or at the input (SSPC 1).
        # This is very impractical and a more viable should eventually be found, but it works for
        # now since all connection are done through the power train builder mechanic, which we
        # can change at will.

        if self.options["at_bus_output"]:
            factor = -1.0
        else:
            factor = 1.0

        outputs["dc_voltage_out"] = (
            inputs["dc_voltage_in"] + factor * inputs["dc_current_in"] * inputs["resistance_sspc"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        if self.options["at_bus_output"]:
            factor = -1.0
        else:
            factor = 1.0

        partials["dc_voltage_out", "dc_voltage_in"] = np.eye(number_of_points)
        partials["dc_voltage_out", "resistance_sspc"] = factor * np.diag(inputs["dc_current_in"])
        partials["dc_voltage_out", "dc_current_in"] = factor * np.diag(inputs["resistance_sspc"])
