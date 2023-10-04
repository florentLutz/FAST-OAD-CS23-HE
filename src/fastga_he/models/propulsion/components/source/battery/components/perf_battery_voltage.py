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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # By default this is the name of the output of this component, however, depending on the
        # mode, we might want to change it
        self.output_name = "voltage_out"

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="direct_bus_connection",
            default=False,
            types=bool,
            desc="If the battery is directly connected to a bus, a special mode is required to "
            "interface the two",
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        if self.options["direct_bus_connection"]:
            self.output_name = "battery_voltage"

        self.add_input("module_voltage", units="V", val=np.full(number_of_points, np.nan))

        self.add_output(self.output_name, units="V", val=np.full(number_of_points, 500.0))

        self.declare_partials(of=self.output_name, wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs[self.output_name] = inputs["module_voltage"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials[self.output_name, "module_voltage"] = np.eye(number_of_points)
