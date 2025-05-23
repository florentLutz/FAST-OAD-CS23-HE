# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDCSSPCOperationalCost(om.ExplicitComponent):
    """
    Computation of the DC SSPC annual operational cost. The lifespan expectancy is obtained from
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
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":purchase_cost",
            val=np.nan,
            units="USD",
            desc="Maximum current flowing through the SSPC",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":lifespan",
            val=10.0,
            units="yr",
            desc="Expected lifespan of the DC SSPC electronic",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":operational_cost",
            val=100.0,
            units="USD/yr",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":operational_cost"] = (
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":purchase_cost"]
            / inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":lifespan"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":operational_cost",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":purchase_cost",
        ] = 1.0 / inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":lifespan"]

        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":operational_cost",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":lifespan",
        ] = (
            -inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":purchase_cost"]
            / inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":lifespan"] ** 2.0
        )
