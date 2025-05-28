# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCTurboshaftCost(om.ExplicitComponent):
    """
    Computation of the turboshaft engine purchase cost from :cite:`gudmundsson:2013`.
    """

    def initialize(self):
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):
        turboshaft_id = self.options["turboshaft_id"]

        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="hp",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":purchase_cost",
            units="USD",
            val=1e4,
            desc="Unit purchase cost of the turboshaft engine"
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        outputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":purchase_cost"] = (
            377.4
            * (
                inputs[
                    "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
                ]
                * inputs["data:cost:cpi_2012"]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":purchase_cost",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = 377.4 * inputs["data:cost:cpi_2012"]

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":purchase_cost",
            "data:cost:cpi_2012",
        ] = (
            377.4
            * inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"]
        )
