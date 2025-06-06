# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCSumOperationalCost(om.ExplicitComponent):
    """
    Computation of summing all the operational cost.
    """

    def initialize(self):
        self.options.declare("cost_components_type", types=list, default=[])
        self.options.declare("cost_components_name", types=list, default=[])

    def setup(self):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        self.add_input(
            "data:cost:operation:annual_crew_cost",
            val=np.nan,
            units="USD/yr",
            desc="Annual flight crew cost of the aircraft",
        )
        self.add_input(
            "data:cost:operation:annual_airport_cost",
            val=np.nan,
            units="USD/yr",
            desc="Annual airport cost of the aircraft",
        )
        self.add_input(
            "data:cost:operation:annual_insurance_cost",
            val=np.nan,
            units="USD/yr",
            desc="Annual insurance cost of the aircraft",
        )
        self.add_input(
            "data:cost:operation:annual_loan_cost",
            val=0.0,
            units="USD/yr",
            desc="Annual loan cost of the aircraft",
        )
        self.add_input(
            "data:cost:operation:miscellaneous_cost",
            val=np.nan,
            units="USD/yr",
            desc="Annual miscellaneous cost per aircraft",
        )
        self.add_input(
            "data:cost:operation:maintenance_cost",
            val=np.nan,
            units="USD/yr",
            desc="Annual maintenance cost per aircraft",
        )
        self.add_input(
            name="data:cost:operation:annual_fuel_cost",
            val=np.nan,
            units="USD/yr",
        )
        self.add_input(
            name="data:cost:operation:annual_electricity_cost",
            val=np.nan,
            units="USD/yr",
        )
        self.add_input(
            "data:cost:operation:additional_cost",
            val=0.0,
            units="USD/yr",
            desc="Yearly additional cost that doesn't considered in this model",
        )

        for component_type, component_name in zip(cost_components_type, cost_components_name):
            self.add_input(
                "data:propulsion:he_power_train:"
                + component_type
                + ":"
                + component_name
                + ":operational_cost",
                units="USD/yr",
                val=np.nan,
            )

        self.add_output("data:cost:operation:annual_cost_per_unit", units="USD/yr", val=0.0)

        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_cost_per_unit"] = np.sum(inputs.values())
