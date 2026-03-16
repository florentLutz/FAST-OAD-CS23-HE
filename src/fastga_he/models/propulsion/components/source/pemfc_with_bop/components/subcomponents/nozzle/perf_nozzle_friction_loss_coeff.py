# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzleFrictionLossCoefficient(om.ExplicitComponent):
    """
    Computation of the friction pressure loss coefficient of the nozzle.
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
            name="nozzle_id",
            default=None,
            desc="Identifier of the nozzle",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "nozzle_darcy_friction_factor",
            val=0.3,
            units="unitless",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":alpha",
            val=np.nan,
            units="rad",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
            units="unitless",
            val=2.0,
        )

        self.add_output(
            "nozzle_friction_loss_coefficient",
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
            wrt="nozzle_darcy_friction_factor",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        darcy_friction_factor = inputs["nozzle_darcy_friction_factor"]
        alpha = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":alpha"
        ]
        area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio"
        ]

        outputs["nozzle_friction_loss_coefficient"] = (
            darcy_friction_factor / (8.0 * np.sin(alpha)) * (1.0 - (1.0 / area_ratio) ** 2.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]
        number_of_points = self.options["number_of_points"]

        darcy_friction_factor = inputs["nozzle_darcy_friction_factor"]
        alpha = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":alpha"
        ]
        area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio"
        ]

        partials["nozzle_friction_loss_coefficient", "nozzle_darcy_friction_factor"] = (
            np.ones(number_of_points) / (8.0 * np.sin(alpha)) * (1.0 - (1.0 / area_ratio) ** 2.0)
        )
        partials[
            "nozzle_friction_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":alpha",
        ] = (
            -darcy_friction_factor
            / (8.0 * np.sin(alpha) ** 2.0)
            * (1.0 - (1.0 / area_ratio) ** 2.0)
            * np.cos(alpha)
        )
        partials[
            "nozzle_friction_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
        ] = darcy_friction_factor / (8.0 * np.sin(alpha)) * (2.0 / area_ratio**3.0)
