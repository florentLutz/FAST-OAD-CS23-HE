# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCurrentRMS1Phase(om.ExplicitComponent):
    """
    Computation of the rms current in one phase based on the current in all phases (DC equivalent).
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_current_rms_in",
            units="A",
            val=np.nan,
            shape=number_of_points,
        )

        self.add_output(
            "ac_current_rms_in_one_phase",
            units="A",
            val= 3.33,
            shape=number_of_points,
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points) / 3.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["ac_current_rms_in_one_phase"] = inputs["ac_current_rms_in"] / 3.0
