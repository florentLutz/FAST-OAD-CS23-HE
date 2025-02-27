# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingPEMFCStackVolume(om.ExplicitComponent):
    """
    Computation of the PEMFC stack volume based on power density, which can be derived from the
    regression model under the methodology directory.
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

        outputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume"] = (
            inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
        ] = (
            -inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
            ]
            ** 2
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
        ] = (
            1
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
            ]
        )
