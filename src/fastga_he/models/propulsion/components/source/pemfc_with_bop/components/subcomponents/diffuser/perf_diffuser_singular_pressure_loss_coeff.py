# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserSingularPressureLossCoefficient(om.ExplicitComponent):
    """
    Computation of the singular pressure loss coefficient of the diffuser.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="diffuser_id",
            default=None,
            desc="Identifier of the diffuser",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":alpha",
            val=np.nan,
            units="rad",
        )

        self.add_output(
            "diffuser_singular_pressure_loss_coefficient",
            val=0.3,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        alpha = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":alpha"
        ]

        if alpha > 0.0:
            outputs["diffuser_singular_pressure_loss_coefficient"] = (
                4.0
                * np.tan(
                    inputs[
                        "data:propulsion:he_power_train:PEMFC_stack_bop:"
                        + pemfc_stack_bop_id
                        + ":"
                        + diffuser_id
                        + ":alpha"
                    ]
                )
                ** 1.25
            )

        else:
            outputs["diffuser_singular_pressure_loss_coefficient"] = 0.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        alpha = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":alpha"
        ]

        partials[
            "diffuser_singular_pressure_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":alpha",
        ] = np.where(alpha > 0.0, 5.0 * np.tan(alpha) ** 0.25 / np.cos(alpha) ** 2.0, 0.0)
