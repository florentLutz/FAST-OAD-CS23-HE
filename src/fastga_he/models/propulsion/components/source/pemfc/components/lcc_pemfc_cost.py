# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCPEMFCStackCost(om.ExplicitComponent):
    """
    Computation of the PEMFC stack purchase cost based on the purchase cost prediction from
    :cite:`fuhren:2022` and the annual delivery prediction from site:
    https://interactanalysis.com/insight/in-2030-over-45000-heavy-trucks-will-run-on-hydrogen-in-europe/.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power of the PEMFC stack has to provide during the mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":purchase_cost",
            units="USD",
            val=1e4,
            desc="Unit purchase cost of the pemfc stack",
        )

        self.declare_partials(of="*", wrt="*", val=65.7)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":purchase_cost"
        ] = (
            65.7
            * inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"]
        )
