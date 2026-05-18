# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class LCCInverterCost(om.ExplicitComponent):
    """
    Computation of inverter purchase cost based on a 3-level NPC architecture from
    :cite:`kremer:2022`. This is estimated based on 6 IGBTs per level, a 20% cost contribution
    of the IGBTs to the total cost based on :cite:`burkart:2013`. The IGBT suggested pricing
    (115 USD) is given by
    https://www.mouser.fr/ProductDetail/Infineon-Technologies/FZ600R12KP4?qs=lxTgnyf4o0eNz5ooVj7tEA%3D%3D.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):
        inverter_id = self.options["inverter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + inverter_id
            + ":number_of_switches",
            val=18.0,
            units="unitless",
            desc="Number of switches in the inverter based on the architecture (e.g., "
            "18 for a 3-level NPC inverter)",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost",
            units="USD",
            val=10350.0,
            desc="Unit purchase cost of the inverter",
        )

        self.declare_partials(of="*", wrt="*", val=575.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost"] = (
            115.0
            * inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + inverter_id
                + ":number_of_switches"
            ]
            / 0.2
        )
