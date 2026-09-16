# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCPowertrainCost(om.ExplicitComponent):
    """
    Computation of summing all the powertrain component cost. This is only for post-processing.
    """

    def initialize(self):
        self.options.declare("cost_components_type", types=list, default=[])
        self.options.declare("cost_components_name", types=list, default=[])

    def setup(self):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        for component_type, component_name in zip(cost_components_type, cost_components_name):
            self.add_input(
                "data:propulsion:he_power_train:"
                + component_type
                + ":"
                + component_name
                + ":purchase_cost",
                units="USD",
                val=np.nan,
            )

        self.add_output(
            "data:cost:production:powertrain_cost_per_unit",
            units="USD",
            val=0.0,
            desc="Unadjusted powertrain cost per unit of the aircraft.",
        )

    def setup_partials(self):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        for component_type, component_name in zip(cost_components_type, cost_components_name):
            self.declare_partials(
                "*",
                "data:propulsion:he_power_train:"
                + component_type
                + ":"
                + component_name
                + ":purchase_cost",
                val=1.0,
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:powertrain_cost_per_unit"] = np.sum(inputs.values())
