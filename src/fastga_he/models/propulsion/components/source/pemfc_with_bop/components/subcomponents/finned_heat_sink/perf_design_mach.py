# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesDesignMach(om.ExplicitComponent):
    """
    Compute the mach number from the true airspeed and sound speed.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input(name="speed_of_sound", units="m/s", val=np.nan, shape=number_of_points)

        self.add_output(name="mach", units="unitless", val=0.28)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            "*",
            "*",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["mach"] = np.max(inputs["true_airspeed"] / inputs["speed_of_sound"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mach = inputs["true_airspeed"] / inputs["speed_of_sound"]
        max_mach = np.max(mach)

        partials["mach", "true_airspeed"] = np.where(
            mach == max_mach, 1.0 / inputs["speed_of_sound"], 1e-6
        )

        partials["mach", "speed_of_sound"] = np.where(
            mach == max_mach, -inputs["true_airspeed"] / inputs["speed_of_sound"] ** 2.0, 1e-6
        )
