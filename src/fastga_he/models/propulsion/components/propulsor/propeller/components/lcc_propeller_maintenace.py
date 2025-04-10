# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class LCCPropellerMaintenance(om.ExplicitComponent):
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
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":constant_speed_prop",
            val=1.0,
            desc="Value set to 1.0 if constant-speed propeller, 0.0 for fixed-pitch propeller",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":turboprop_connection",
            val=0.0,
            desc="Value set to 1.0 if powered by turboprop, 0.0 for other type of engine",
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":maintenance_per_unit",
            units="USD/yr",
            val=500.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        f_constant_speed = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":constant_speed_prop"
        ]
        turboprop_connection = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":turboprop_connection"
        ]

        outputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":maintenance_per_unit"
        ] = (1.0 - f_constant_speed) * 147.0 + f_constant_speed * (
            517.0 + turboprop_connection * 383.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]
        f_constant_speed = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":constant_speed_prop"
        ]
        turboprop_connection = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":turboprop_connection"
        ]

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":maintenance_per_unit",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":constant_speed_prop",
        ] = -147.0 + (517.0 + turboprop_connection * 383.0)

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":maintenance_per_unit",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":turboprop_connection",
        ] = 383.0 * f_constant_speed
