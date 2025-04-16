# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCSumProductionCost(om.ExplicitComponent):
    """
    Computation of summing all the costs and reductions for both production phase.
    """

    def initialize(self):
        self.options.declare("cost_components_type", types=list, default=[])
        self.options.declare("cost_components_name", types=list, default=[])

    def setup(self):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        self.add_input(
            "data:cost:production:engineering_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Engineering adjusted cost per aircraft",
        )

        self.add_input(
            "data:cost:production:tooling_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Tooling adjusted cost per aircraft",
        )

        self.add_input(
            "data:cost:production:manufacturing_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Manufacturing adjusted cost per aircraft",
        )

        self.add_input(
            "data:cost:production:flight_test_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Development flight test adjusted cost per aircraft",
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
            desc="Development flight test adjusted cost per aircraft",
        )

        self.add_input(
            "data:cost:production:dev_support_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Development support adjusted cost per aircraft",
        )

        self.add_input(
            "data:cost:production:avionics_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Avionics adjusted cost per aircraft",
        )

        self.add_input(
            "data:cost:production:certification_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Certification adjusted cost per aircraft",
        )

        self.add_input(
            "data:cost:production:landing_gear_cost_reduction",
            val=np.nan,
            units="USD",
            desc="Cost reduction if fixed landing gear design is selected",
        )

        for component_type, component_name in zip(cost_components_type, cost_components_name):
            self.add_input(
                "data:propulsion:he_power_train:"
                + component_type
                + ":"
                + component_name
                + ":cost_per_unit",
                units="USD",
                val=np.nan,
            )

        self.add_output("data:cost:production_cost_per_unit", units="USD", val=0.0)

        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production_cost_per_unit"] = np.sum(inputs.values())
