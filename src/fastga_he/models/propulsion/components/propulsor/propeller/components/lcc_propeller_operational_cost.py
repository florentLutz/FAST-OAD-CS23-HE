# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class LCCPropellerOperationalCost(om.ExplicitComponent):
    """
    Computation of the propeller maintenance cost based on the mean service price from
    https://aircraftaccessoriesofok.com/aircraft-propeller-overhaul-cost/.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]

        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":type",
            val=1.0,
            desc="Value set to 1.0 if constant-speed propeller, 0.0 for fixed-pitch propeller. "
            "This is only use in cost estimation, does not affect other propeller-related "
            "models.",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":turboprop_connection",
            val=0.0,
            desc="Value set to 1.0 if powered by turboprop, 0.0 for other type of engine",
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":operational_cost",
            units="USD/yr",
            val=500.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        prop_type = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":type"]
        turboprop_connection = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":turboprop_connection"
        ]

        outputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":operational_cost"
        ] = (1.0 - prop_type) * 147.0 + prop_type * (517.0 + turboprop_connection * 383.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]
        prop_type = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":type"]
        turboprop_connection = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":turboprop_connection"
        ]

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":operational_cost",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":type",
        ] = -147.0 + (517.0 + turboprop_connection * 383.0)

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":operational_cost",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":turboprop_connection",
        ] = 383.0 * prop_type
