# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om


class LCCDCDCConverterCost(om.ExplicitComponent):
    """
    Computation of convertor purchase cost based on a 3-level Buck-Boost architecture from
    :cite:`menzi:2024`. This is estimated based on two IGBTs per switch, a 20% cost contribution
    of the IGBTs to the total cost based on :cite:`burkart:2013`. The IGBT suggested pricing
    (115 USD) is given by
    https://www.mouser.fr/ProductDetail/Infineon-Technologies/FZ600R12KP4?qs=lxTgnyf4o0eNz5ooVj7tEA%3D%3D.
    """

    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":number_of_switches",
            val=4.0,
            units="unitless",
            desc="Number of switches in the converter based on the architecture (e.g., "
            "4 for a 3-level Buck-Boost converter)",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":purchase_cost",
            units="USD",
            val=4600.0,
            desc="Unit purchase cost of the converter",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", val=1150.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":purchase_cost"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":number_of_switches"
            ]
            * 115.0
            * 2.0
            / 0.2
        )
