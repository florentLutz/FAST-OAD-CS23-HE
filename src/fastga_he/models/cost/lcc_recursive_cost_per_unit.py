# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCRecursiveCost(om.ExplicitComponent):
    """
    Computation of summing all the recursive cost per unit.
    """

    def initialize(self):
        self.options.declare("cost_components_type", types=list, default=[])
        self.options.declare("cost_components_name", types=list, default=[])

    def setup(self):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        self.add_input(
            "data:cost:production:manufacturing_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Manufacturing adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:production:quality_control_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Quality control adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:production:material_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Material adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:production:avionics_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Avionics adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:production:landing_gear_cost_reduction",
            val=0.0,
            units="USD",
            desc="Cost reduction if fixed landing gear design is selected",
        )

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

        self.add_output("data:cost:recursive_cost_per_unit", units="USD", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:recursive_cost_per_unit"] = np.sum(inputs.values())
