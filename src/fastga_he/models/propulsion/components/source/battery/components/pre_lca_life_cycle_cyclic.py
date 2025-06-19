#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCABatteryCyclicAging(om.ExplicitComponent):
    """
    Computation of the number of cycle necessary to reach a nominal capacity loss of 40% (which
    seems to be the standard for the reference cell). This will be treated as the lifespan of the
    cell.
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
            + ":aging:cyclic_effect_DOD",
            units="unitless",
            val=np.nan,
            desc="Multiplicative factor for the effect of the DOD of one cycle on cyclic aging",
        )
        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_k_factor",
            units="unitless",
            val=2.98,
            desc="Corrective factor to adjust model to manufacturer's data. Default value leads to "
            "a loss of 40% of capacity after 500 cycle, a DOD of 100% under 23 degC",
        )
        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":average_cell_temperature",
            val=np.nan,
            units="degK",
            desc="Time step averaged temperature of the cell during the mission",
        )

        self.add_output(
            name="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            val=500.0,
            units="unitless",
            desc="Expected lifetime of the battery pack, expressed in cycles. Default value is the "
            "number of cycle required for the reference cell to reach 60% nominal capacity",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        f_dod = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_DOD"
        ]
        avg_temperature = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":average_cell_temperature"
        ]
        k_factor = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_k_factor"
        ]

        life_cycle = (0.4 / f_dod / np.exp(-4345 / avg_temperature) / k_factor) ** 2.0

        outputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"] = (
            life_cycle
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        f_dod = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_DOD"
        ]
        avg_temperature = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":average_cell_temperature"
        ]
        k_factor = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_k_factor"
        ]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_DOD",
        ] = -2.0 * (0.4 / np.exp(-4345 / avg_temperature) / k_factor) ** 2.0 / f_dod**3.0
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":aging:cyclic_effect_k_factor",
        ] = -2.0 * (0.4 / np.exp(-4345 / avg_temperature) / f_dod) ** 2.0 / k_factor**3.0
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":average_cell_temperature",
        ] = (
            -2.0
            * (0.4 / k_factor / f_dod) ** 2.0
            / np.exp(-4345 / avg_temperature) ** 3.0
            * np.exp(-4345 / avg_temperature)
            * 4345.0
            / avg_temperature**2.0
        )
