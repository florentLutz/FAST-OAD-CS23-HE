# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCTurboshaftCost(om.ExplicitComponent):
    """
    Computation of the turboshaft engine purchasing cost from :cite:`gudmundsson:2013`.
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
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":max_shaft_power",
            units="hp",
            val=np.nan,
            desc="Maximum shaft power the turboshaft has to provide",
        )

        self.add_output(
            name="data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":cost_per_engine",
            units="USD",
            val=1e4,
            desc="Cost of the turboshaft per unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        outputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":cost_per_engine"
        ] = 377.4 * (
            inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":max_shaft_power"
            ]
            * inputs["data:cost:cpi_2012"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":cost_per_engine",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":max_shaft_power",
        ] = 377.4 * inputs["data:cost:cpi_2012"]
        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":cost_per_engine",
            "data:cost:cpi_2012",
        ] = (
            377.4
            * inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":max_shaft_power"
            ]
        )
