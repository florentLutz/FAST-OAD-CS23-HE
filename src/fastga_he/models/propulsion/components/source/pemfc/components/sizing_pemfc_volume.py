# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingPEMFCStackVolume(om.ExplicitComponent):
    """
    Computation of the PEMFC stack volume based on power density.
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
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
            units="kW/m**3",
            val=np.nan,
        )

        self.add_input(
            "settings:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":k_volume",
            val=1.0,
            desc="A tuning factor allows the PEMFC stack volume to be changed manually.",
        )

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power of the PEMFC stack has to provide during the mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            units="m**3",
            val=5.0,
            desc="Volume of the PEMFC stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        k_volume = inputs[
            "settings:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":k_volume"
        ]
        power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"
        ]
        power_density = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
        ]

        outputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume"] = (
            k_volume * power_max / power_density
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        k_volume = inputs[
            "settings:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":k_volume"
        ]
        power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"
        ]
        power_density = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
        ] = -k_volume * power_max / power_density**2

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
        ] = k_volume / power_density

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            "settings:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":k_volume",
        ] = power_max / power_density
