# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCSCost(om.ExplicitComponent):
    """
    Computation of summing all the costs and reductions for both production and operation phase.
    """

    def initialize(self):
        self.options.declare("input_costs", types=list, default=[])

    def setup(self):
        for cost in self.options["input_costs"]:
            self.add_input(cost, units="USD", val=0.0)

        self.add_output("data:cost:production_cost_per_unit", units="USD", val=0.0)

        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["sum"] = np.sum(inputs.values())
