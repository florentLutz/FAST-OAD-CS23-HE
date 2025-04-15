# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesEngineRPM(om.ExplicitComponent):
    """
    Computation of the engine rpm. It differs from the shaft rm because this type of engine
    has a built-in reduction ratio of 2.43 as per Rotax 912 operation manual.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output("engine_rpm", units="min**-1", val=5500.0, shape=number_of_points)

        self.declare_partials(of="engine_rpm", wrt="rpm", val=2.43)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["engine_rpm"] = 2.43 * inputs["rpm"]
