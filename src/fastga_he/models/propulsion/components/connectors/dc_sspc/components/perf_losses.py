# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDCSSPCLosses(om.ExplicitComponent):
    """
    This module computes the total losses inside the SSPC module. As the SSSD is responsible for
    the transmission of power :cite:`liu:2012`, we will assume that only its losses will be relevant
    to compute. We will also assume that the power losses corresponds to the switching losses in
    the on state of the SSSD.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "closed",
            default=True,
            desc="Boolean to choose whether the breaker is closed or " "not.",
            types=bool,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_current_in",
            val=np.full(number_of_points, np.nan),
            units="A",
        )
        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
        )
        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
        )

        self.add_output(
            "power_losses",
            val=np.full(number_of_points, 0.0),
            units="W",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]

        # We may encounter a problem with the computation of losses if the SSPC is open,
        # because theoretically resistance is infinite while current is nil. Common sense dictate
        # that the losses are nil in this which is what we will output

        if not self.options["closed"]:

            outputs["power_losses"] = np.zeros(number_of_points)

        else:

            outputs["power_losses"] = (
                np.abs(inputs["dc_voltage_in"] - inputs["dc_voltage_out"]) * inputs["dc_current_in"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        if not self.options["closed"]:

            partials["power_losses", "dc_current_in"] = np.zeros(number_of_points)
            partials["power_losses", "dc_voltage_in"] = np.zeros(number_of_points)
            partials["power_losses", "dc_voltage_out"] = np.zeros(number_of_points)

        else:
            partials["power_losses", "dc_current_in"] = np.abs(
                inputs["dc_voltage_in"] - inputs["dc_voltage_out"]
            )
            partials["power_losses", "dc_voltage_in"] = (
                np.sign(inputs["dc_voltage_in"] - inputs["dc_voltage_out"])
                * inputs["dc_current_in"]
            )
            partials["power_losses", "dc_voltage_out"] = (
                -np.sign(inputs["dc_voltage_in"] - inputs["dc_voltage_out"])
                * inputs["dc_current_in"]
            )
