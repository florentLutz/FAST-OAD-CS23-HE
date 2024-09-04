# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTorque(om.ExplicitComponent):
    """
    Computation of the torque at the input of the turbo generator based on the rpm and shaft
    power in constant.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("shaft_power_in", units="W", val=np.nan, shape=number_of_points)

        self.add_output("torque_in", units="N*m", val=400.0, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        omega = inputs["rpm"] * 2.0 * np.pi / 60

        outputs["torque_in"] = inputs["shaft_power_in"] / omega

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        omega = inputs["rpm"] * 2.0 * np.pi / 60

        partials["torque_in", "shaft_power_in"] = 1.0 / omega
        partials["torque_in", "rpm"] = -inputs["shaft_power_in"] / omega**2.0 * 2.0 * np.pi / 60
