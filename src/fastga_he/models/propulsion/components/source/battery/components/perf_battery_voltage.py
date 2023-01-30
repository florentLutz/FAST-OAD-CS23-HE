# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesBatteryVoltage(om.ExplicitComponent):
    """
    Computation of the voltage at the output of the battery, assumes for now that it is equal to
    the voltage output of the modules. May change in the future hence why it is in a separate
    module.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("module_voltage", units="V", val=np.full(number_of_points, np.nan))

        self.add_output("voltage_out", units="V", val=np.full(number_of_points, 500.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["voltage_out"] = inputs["module_voltage"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["voltage_out", "module_voltage"] = np.eye(number_of_points)
