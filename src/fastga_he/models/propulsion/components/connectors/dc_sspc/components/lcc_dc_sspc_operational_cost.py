# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDCSSPCOperationalCost(om.ExplicitComponent):
    """
    Computation of the SSPC annual operational cost. The lifespan expectancy is obtained from
    :cite:`cao:2023`.
    """

    def initialize(self):
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

    def setup(self):
        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Maximum current flowing through the SSPC",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":operational_cost",
            val=100.0,
            units="USD/yr",
        )

        self.declare_partials(of="*", wrt="*", val=0.1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":operational_cost"] = (
            0.1 * inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":cost_per_unit"]
        )
