# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCICECost(om.ExplicitComponent):
    """
    Computation of ICE engine purchase cost from :cite:`gudmundsson:2013`.
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
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
            units="hp",
            val=np.nan,
            desc="Maximum power the motor has to provide at Sea Level",
        )
        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ICE:" + ice_id + ":purchase_cost",
            units="USD",
            val=1e4,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ice_id = self.options["ice_id"]

        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":purchase_cost"] = (
            174.0
            * inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL"]
            * inputs["data:cost:cpi_2012"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ice_id = self.options["ice_id"]

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":purchase_cost",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
        ] = 174.0 * inputs["data:cost:cpi_2012"]

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":purchase_cost",
            "data:cost:cpi_2012",
        ] = 174.0 * inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL"]
