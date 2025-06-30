#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCABatteryCalendarAging(om.ExplicitComponent):
    """
    Computation of the capacity lost to calendar aging after a certain number of cycles. Adaptation
    of the model from :cite:`chen:2019`.
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
            name="number_of_cycles",
            val=np.nan,
            units="unitless",
            desc="Number of cycle at which to evaluate capacity loss",
        )
        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":time_between_cycle",
            units="d",  # This is the unit for days
            val=np.nan,
            desc="Amount of time between two cycles of the battery, used for calendar aging computation",
        )
        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_SOC",
            units="unitless",
            val=np.nan,
            desc="Multiplicative factor for the effect of storage SOC on calendar aging",
        )
        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_k_factor",
            units="unitless",
            val=1.0,
            desc="Corrective factor to adjust calendar aging model.",
        )
        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:storage_temperature",
            units="degK",
            val=298.15,
            desc="Temperature at which the battery is stored, default is 25 degC",
        )

        self.add_output(
            name="capacity_loss_calendar",
            val=0.0,
            units="unitless",
            desc="Capacity lost due to calendar aging",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        f_soc = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_SOC"
        ]
        storage_temperature = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:storage_temperature"
        ]
        k_factor = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_k_factor"
        ]
        # Safeguard
        n_cycles = np.maximum(inputs["number_of_cycles"], 100)
        time_btw_cycle = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":time_between_cycle"
        ]

        relative_capacity_loss = (
            k_factor
            * f_soc
            * np.exp(-3053 / storage_temperature)
            * (n_cycles * time_btw_cycle) ** 0.5
        )

        outputs["capacity_loss_calendar"] = relative_capacity_loss

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        f_soc = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_SOC"
        ]
        storage_temperature = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:storage_temperature"
        ]
        k_factor = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_k_factor"
        ]
        n_cycles = np.maximum(inputs["number_of_cycles"], 100)
        time_btw_cycle = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":time_between_cycle"
        ]

        partials[
            "capacity_loss_calendar",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_SOC",
        ] = k_factor * np.exp(-3053 / storage_temperature) * (n_cycles * time_btw_cycle) ** 0.5
        partials[
            "capacity_loss_calendar",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:storage_temperature",
        ] = (
            k_factor
            * f_soc
            * np.exp(-3053 / storage_temperature)
            * (n_cycles * time_btw_cycle) ** 0.5
            * 3053
            / storage_temperature**2.0
        )
        partials[
            "capacity_loss_calendar",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:calendar_effect_k_factor",
        ] = f_soc * np.exp(-3053 / storage_temperature) * (n_cycles * time_btw_cycle) ** 0.5
        partials["capacity_loss_calendar", "number_of_cycles"] = np.where(
            n_cycles == inputs["number_of_cycles"],
            0.5
            * k_factor
            * f_soc
            * np.exp(-3053 / storage_temperature)
            * (time_btw_cycle / n_cycles) ** 0.5,
            1e-6,
        )
        partials[
            "capacity_loss_calendar",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":time_between_cycle",
        ] = (
            0.5
            * k_factor
            * f_soc
            * np.exp(-3053 / storage_temperature)
            * (n_cycles / time_btw_cycle) ** 0.5
        )
