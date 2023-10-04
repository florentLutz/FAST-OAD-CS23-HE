# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesBatteryDirectBusConnection(om.ImplicitComponent):
    """
    In case we attempt to directly plug the battery into a bus, a small change to its
    inputs/outputs will be required. This is what this component does.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        # This input will come from the bus
        self.add_input("voltage_out", units="V", val=np.full(number_of_points, np.nan))
        # This one because the current imposes a voltage
        self.add_input("battery_voltage", units="V", val=np.full(number_of_points, np.nan))

        self.add_output(
            "dc_current_out",
            val=np.full(number_of_points, 400.0),
            units="A",
            desc="Current at the output side of the battery",
            lower=-1000.0,
            upper=1000.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(self, inputs, outputs, residuals):

        residuals["dc_current_out"] = inputs["voltage_out"] - inputs["battery_voltage"]

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        partials["dc_current_out", "voltage_out"] = np.eye(number_of_points)
        partials["dc_current_out", "battery_voltage"] = -np.eye(number_of_points)
