#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCABatteryCalendarAgingSOCEffect(om.ExplicitComponent):
    """
    Computation of the effect of the SOC at which the battery is stored on the aging of the battery.
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
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:storage_SOC",
            units="percent",
            val=100.0,
            desc="State of charge at which the battery is stored, default is 100.0 like a flight-ready battery",
        )

        self.add_output(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_SOC",
            units="unitless",
            val=100.0,
            desc="Multiplicative factor for the effect of storage SOC on calendar aging",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_SOC",
            wrt="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:storage_SOC",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        # As a safeguard
        storage_soc = np.clip(
            inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":aging:storage_SOC"
            ],
            20.0,
            100.0,
        )

        f_soc = (
            0.0007459 * storage_soc**3.0 - 0.1751 * storage_soc**2.0 + 12.08 * storage_soc - 103.5
        )

        outputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_SOC"
        ] = f_soc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        # As a safeguard
        storage_soc = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":aging:storage_SOC"
        ]

        if storage_soc > 100.0 or storage_soc < 20.0:
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":aging:calendar_effect_SOC",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":aging:storage_SOC",
            ] = 1e-6
        else:
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":aging:calendar_effect_SOC",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":aging:storage_SOC",
            ] = 3.0 * 0.0007459 * storage_soc**2.0 - 2.0 * 0.1751 * storage_soc + 12.08
