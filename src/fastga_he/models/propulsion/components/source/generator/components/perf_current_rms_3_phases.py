# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCurrentRMS3Phases(om.ExplicitComponent):
    """
    Computation of the current in all 3 phases of the generator based on the current in one phase
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_current_rms_out_one_phase",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current at the output side in one phase of the generator",
        )

        self.add_output(
            "ac_current_rms_out",
            val=np.full(number_of_points, 600.0),
            units="A",
            desc="Current at the output side of the generator",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=3.0 * np.ones(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["ac_current_rms_out"] = 3.0 * inputs["ac_current_rms_out_one_phase"]
