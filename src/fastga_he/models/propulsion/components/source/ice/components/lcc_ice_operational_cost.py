# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCICEOperationalCost(om.ExplicitComponent):
    """
    Computation of ICE engine annual operational cost from
    http://blog.overhaulbids.com/lycoming-overhaul-cost/.
    """

    def initialize(self):
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):
        ice_id = self.options["ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume",
            units="inch**3",
            val=np.nan,
        )
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ICE:" + ice_id + ":operational_cost",
            units="USD/yr",
            val=1e4,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ice_id = self.options["ice_id"]

        volume = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume"]
        flight_hour = inputs["data:TLAR:flight_hours_per_year"]

        v_clipped = np.clip(volume, 42.8156, None)

        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":operational_cost"] = (
            (0.103 * v_clipped - 4.41) * flight_hour / 1.8
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ice_id = self.options["ice_id"]

        volume = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume"]
        flight_hour = inputs["data:TLAR:flight_hours_per_year"]

        v_clipped = np.clip(volume, 42.8156, None)

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":operational_cost",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume",
        ] = np.where(volume == v_clipped, 0.103 * flight_hour / 1.8, 1e-6)

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = (0.103 * v_clipped - 4.41) / 1.8
