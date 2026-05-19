# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om


class LCCRectifierCost(om.ExplicitComponent):
    """
    Computation of the rectifier purchase cost based on three-phase full bridge architecture.
     This is estimated based on two diodes per branch, a 20% cost contribution
    of the main electronics to the total cost based on :cite:`burkart:2013`. The IGBT suggested pricing
    (115 USD) is given by
    https://www.mouser.fr/ProductDetail/Infineon-Technologies/FZ600R12KP4?qs=lxTgnyf4o0eNz5ooVj7tEA%3D%3D.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":number_of_diodes",
            val=6.0,
            units="unitless",
            desc="Number of diodes in the rectifier based on the architecture (e.g., "
            "6 for a three-phase full bridge rectifier)",
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost",
            units="USD",
            val=3450.0,
            desc="Unit purchase cost of the rectifier",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", val=575.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost"] = (
            115.0
            * inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":number_of_diodes"
            ]
            / 0.2
        )
