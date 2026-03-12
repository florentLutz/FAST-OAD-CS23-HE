# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserFrictionLossCoefficient(om.ExplicitComponent):
    """
    Computation of the friction loss coefficient of the diffuser.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
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
        number_of_points = self.options["number_of_points"]
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
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":beta",
            val=np.nan,
            units="rad",
        )
        self.add_input(
            "diffuser_darcy_friction_factor",
            val=np.nan,
            units="unitless",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio",
            val=np.nan,
            units="unitless",
        )

        self.add_output(
            "diffuser_friction_loss_coefficient",
            val=0.3,
            units="unitless",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="diffuser_darcy_friction_factor",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

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
        beta = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":beta"
        ]
        darcy_friction_factor = inputs["diffuser_darcy_friction_factor"]
        area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio"
        ]

        outputs["diffuser_friction_loss_coefficient"] = (
            darcy_friction_factor
            * (1.0 - area_ratio**2.0)
            / 16.0
            * (1.0 / np.sin(alpha) + 1.0 / np.sin(beta))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]
        number_of_points = self.options["number_of_points"]

        alpha = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":alpha"
        ]
        beta = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":beta"
        ]
        darcy_friction_factor = inputs["diffuser_darcy_friction_factor"]
        area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio"
        ]

        partials["diffuser_friction_loss_coefficient", "diffuser_darcy_friction_factor"] = (
            np.ones(number_of_points)
            * (1.0 - area_ratio**2.0)
            / 16.0
            * (1.0 / np.sin(alpha) + 1.0 / np.sin(beta))
        )

        partials[
            "diffuser_friction_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":alpha",
        ] = (
            -darcy_friction_factor
            * (1.0 - area_ratio**2.0)
            / 16.0
            * (1.0 / np.sin(alpha) ** 2.0)
            * np.cos(alpha)
        )

        partials[
            "diffuser_friction_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":beta",
        ] = (
            -darcy_friction_factor
            * (1.0 - area_ratio**2.0)
            / 16.0
            * (1.0 / np.sin(beta) ** 2.0)
            * np.cos(beta)
        )

        partials[
            "diffuser_friction_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio",
        ] = -darcy_friction_factor * area_ratio / 8.0 * (1.0 / np.sin(alpha) + 1.0 / np.sin(beta))
