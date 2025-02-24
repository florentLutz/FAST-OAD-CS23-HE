# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


MAX_PEMFC_SPECIFIC_POWER = 2.06  # [kW/kg]


class SizingPEMFCStackSpecificPower(om.ExplicitComponent):
    """
    Computation of the maximum specific power provide of PEMFC system exclude the inlet
    compressor. Applied in weight calculation.
    Source: https://www.h3dynamics.com/_files/ugd/3029f7_5111f6ea97244ed09b72a916a8997773.pdf
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power to PEMFC during the mission",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":specific_power",
            units="kW/kg",
            val=0.3,
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_specific_power = 0.0845 * np.log(power_max) + 0.6037

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":specific_power"
        ] = np.clip(unclipped_specific_power, 0.05, MAX_PEMFC_SPECIFIC_POWER)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_specific_power = 0.0845 * np.log(power_max) + 0.6037

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":specific_power",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
        ] = np.where(
            unclipped_specific_power <= MAX_PEMFC_SPECIFIC_POWER
            and unclipped_specific_power >= 0.05,
            0.0845 / power_max,
            1e-6,
        )
