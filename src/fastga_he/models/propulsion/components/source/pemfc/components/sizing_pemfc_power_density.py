# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import logging

_LOGGER = logging.getLogger(__name__)

MAX_PEMFC_POWER_DENSITY = 500  # [kW/m^3]


class SizingPEMFCStackPowerDensity(om.ExplicitComponent):
    """
    Computation of the maximum power density of the PEMFC system excluding the inlet compressor.
    Applied in volume calculation. The data and the regression model of this calculation can be
    found in the methodology folder.
    Source: https://www.h3dynamics.com/_files/ugd/3029f7_5111f6ea97244ed09b72a916a8997773.pdf
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
            name="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power of the PEMFC stack has to provide during the mission",
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

        if unclipped_power_density < 70:
            _LOGGER.info(
                msg="Power density clipped at 70 [kW/m^3] to prevent unrealistic dimension."
            )

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
        ] = np.clip(unclipped_power_density, 70.0, MAX_PEMFC_POWER_DENSITY)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_power_density = 19.816 * np.log(power_max) + 236.48

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
        ] = np.where(
            unclipped_power_density <= MAX_PEMFC_POWER_DENSITY and unclipped_power_density >= 70.0,
            19.816 / power_max,
            1e-6,
        )
