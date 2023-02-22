# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDCSSPCCurrent(om.ExplicitComponent):
    """
    This step may look trivial but it looks necessary to ensure that the connection of components is
    done properly. Indeed bus have current as an input, while harness have it as an output.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_current_out",
            val=np.full(number_of_points, np.nan),
            units="A",
        )

        self.add_output(
            "dc_current_in",
            val=np.full(number_of_points, 550.0),
            units="A",
        )

        self.declare_partials(of="*", wrt="*", val=np.eye(number_of_points))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["dc_current_in"] = inputs["dc_current_out"]
