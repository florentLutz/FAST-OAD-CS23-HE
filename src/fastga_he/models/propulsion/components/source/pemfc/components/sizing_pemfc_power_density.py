# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

MAX_PEMFC_POWER_DENSITY = 500  # [kW/m^3]


class SizingPEMFCStackPowerDensity(om.ExplicitComponent):
    """
    Computation of the maximum power density of PEMFC system excluding the inlet compressor.
    Applied in volume calculation.
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
            name="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
            units="kW/m**3",
            val=250.0,
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

        unclipped_power_density = 19.816 * np.log(power_max) + 236.48

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
        ] = np.clip(unclipped_power_density, 230.0, MAX_PEMFC_POWER_DENSITY)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_power_density = 19.816 * np.log(power_max) + 236.48
        if unclipped_power_density <= MAX_PEMFC_POWER_DENSITY and unclipped_power_density >= 230.0:
            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            ] = 19.816 / power_max
        else:
            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            ] = 0.0
