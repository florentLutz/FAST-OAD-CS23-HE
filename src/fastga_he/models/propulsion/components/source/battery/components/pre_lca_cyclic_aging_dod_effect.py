#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCABatteryCyclicAgingDODEffect(om.ExplicitComponent):
    """
    Computation of the effect of the DOD of the battery on cycle on the aging of the battery.
    Model taken from :cite:`chen:2019`.
    """

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":DOD_main_route",
            units="percent",
            val=np.nan,
            desc="Depth of battery discharge on one cycle",
        )

        self.add_output(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_DOD",
            units="unitless",
            val=10000.0,
            desc="Multiplicative factor for the effect of the DOD of one cycle on cyclic aging",
        )

    def setup_partials(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_DOD",
            wrt="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":DOD_main_route",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        dod = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":DOD_main_route"
        ]

        f_dod = -0.002315 * dod**3.0 + 1.071 * dod**2.0 - 27.49 * dod + 8473

        outputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_DOD"
        ] = f_dod

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        dod = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":DOD_main_route"
        ]

        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_DOD",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":DOD_main_route",
        ] = -3.0 * 0.002315 * dod**2.0 + 2.0 * 1.071 * dod - 27.49
