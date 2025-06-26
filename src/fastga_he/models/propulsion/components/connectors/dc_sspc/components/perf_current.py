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
        self.options.declare(
            "closed",
            default=True,
            desc="Boolean to choose whether the breaker is closed or not.",
            types=bool,
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
            val=np.full(number_of_points, 400.0),
            units="A",
            lower=-2000.0,
            upper=2000.0,
        )

        if self.options["closed"]:
            self.declare_partials(
                of="*",
                wrt="*",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
                val=np.ones(number_of_points),
            )
        else:
            self.declare_partials(
                of="*",
                wrt="*",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
                val=np.full(number_of_points, 1e-6),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]

        if self.options["closed"]:
            outputs["dc_current_in"] = inputs["dc_current_out"]
        else:
            outputs["dc_current_in"] = np.zeros(number_of_points)
