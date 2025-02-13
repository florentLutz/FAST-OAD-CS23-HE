# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


MAX_PEMFC_SYSTEM_SPECIFIC_POWER = 2.06  # kW/kg
MAX_PEMFC_STACK_SPECIFIC_POWER = 4.5  # kW/kg


class PerformancesPEMFCMaxSpecificPowerFuelCellSystem(om.ExplicitComponent):
    # TODO:Proper citation after rebase
    """
    Computation of the sizing specific power provide of PEMFC system exclude the inlet compressor.
    Applied in weight calculation. Source: H3D_H2_UnmannedAviation_Brochure 2024.pptx
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
            name="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power to the pemfc during the mission",
        )

        self.add_output(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":max_specific_power",
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
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_specific_power = 0.0845 * np.log(power_max) + 0.6037

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":max_specific_power"
        ] = np.clip(unclipped_specific_power, 0.05, MAX_PEMFC_SYSTEM_SPECIFIC_POWER)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        power_max = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_specific_power = 0.0845 * np.log(power_max) + 0.6037
        if (
            unclipped_specific_power <= MAX_PEMFC_SYSTEM_SPECIFIC_POWER
            and unclipped_specific_power >= 0.05
        ):
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_specific_power",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = 0.0845 / power_max
        else:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_specific_power",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = 0.0


class PerformancesPEMFCMaxSpecificPowerFuelCellStack(om.ExplicitComponent):
    """
    Computation of the max specific power of PEMFC stack. Applied in weight calculation
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
            name="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power to the pemfc during the mission",
        )

        self.add_output(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":max_specific_power",
            units="kW/kg",
            val=1.0,
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        power_max = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_specific_power = 0.27775 * np.log(power_max) + 1.598

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":max_specific_power"
        ] = np.clip(unclipped_specific_power, 0.05, MAX_PEMFC_STACK_SPECIFIC_POWER)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        unclipped_specific_power = (
            0.27775
            * np.log(
                inputs[
                    "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
                ]
            )
            + 1.598
        )

        if (
            unclipped_specific_power <= MAX_PEMFC_STACK_SPECIFIC_POWER
            and unclipped_specific_power >= 0.05
        ):
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_specific_power",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = (
                0.27775
                / inputs[
                    "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
                ]
            )
        else:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_specific_power",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = 0.0
