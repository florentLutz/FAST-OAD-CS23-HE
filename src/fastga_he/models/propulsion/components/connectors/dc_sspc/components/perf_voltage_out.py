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
        self.options.declare(
            "closed",
            default=True,
            desc="Boolean to choose whether the breaker is closed or not.",
            types=bool,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "efficiency",
            val=np.full(number_of_points, 0.99),
            desc="Value of the SSPC efficiency, assumed constant during operations (eases convergence)",
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

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

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

        if self.options["closed"]:
            if self.options["at_bus_output"]:
                outputs["dc_voltage_out"] = inputs["dc_voltage_in"] * inputs["efficiency"]
            else:
                outputs["dc_voltage_out"] = inputs["dc_voltage_in"] / inputs["efficiency"]
        else:

            # If we start from the principle that there will be a logic for the opening of SSPC (
            # which for instance will open both SSPC at the side of a cable if either one side
            # was to fail or if the cable was to fail, to ensure that no current flows through
            # the cable we must put the same value at each side of said cable (so it also assume
            # that there is a SSPC at both side of the cable)
            outputs["dc_voltage_out"] = inputs["dc_voltage_in"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        if self.options["closed"]:
            if self.options["at_bus_output"]:
                partials["dc_voltage_out", "dc_voltage_in"] = inputs["efficiency"]
                partials["dc_voltage_out", "efficiency"] = inputs["dc_voltage_in"]
            else:
                partials["dc_voltage_out", "dc_voltage_in"] = 1.0 / inputs["efficiency"]
                partials["dc_voltage_out", "efficiency"] = (
                    -inputs["dc_voltage_in"] / inputs["efficiency"] ** 2.0
                )

        else:
            partials["dc_voltage_out", "dc_voltage_in"] = np.ones(number_of_points)
