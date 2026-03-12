# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserExpansionLossCoefficient(om.ExplicitComponent):
    """
    Computation of the geometry expansion loss coefficient of the diffuser.
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
            + ":area_ratio",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "diffuser_singular_pressure_loss_coefficient",
            val=np.nan,
            units="unitless",
        )

        self.add_output(
            "diffuser_expansion_loss_coefficient",
            val=0.3,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio"
        ]
        diffuser_singular_pressure_loss_coefficient = inputs[
            "diffuser_singular_pressure_loss_coefficient"
        ]

        outputs["diffuser_expansion_loss_coefficient"] = (
            diffuser_singular_pressure_loss_coefficient * (1.0 - area_ratio) ** 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio"
        ]
        diffuser_singular_pressure_loss_coefficient = inputs[
            "diffuser_singular_pressure_loss_coefficient"
        ]

        partials[
            "diffuser_expansion_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio",
        ] = -2.0 * diffuser_singular_pressure_loss_coefficient * (1.0 - area_ratio)

        partials[
            "diffuser_expansion_loss_coefficient", "diffuser_singular_pressure_loss_coefficient"
        ] = (1.0 - area_ratio) ** 2.0
