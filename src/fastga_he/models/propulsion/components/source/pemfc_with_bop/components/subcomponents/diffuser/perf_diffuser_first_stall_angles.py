# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserFirstStallAngel(om.ExplicitComponent):
    """
    Computation of the first stall angle of the diffuser.
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
            + ":length_height_ratio",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":length_width_ratio",
            val=np.nan,
            units="unitless",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":first_stall_alpha",
            val=5.0,
            units="deg",
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":first_stall_beta",
            val=5.0,
            units="deg",
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":first_stall_alpha",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":length_height_ratio",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":first_stall_beta",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":length_width_ratio",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        length_height_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":length_height_ratio"
        ]
        length_width_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":length_width_ratio"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":first_stall_alpha"
        ] = 15.1735 * (length_height_ratio**-0.618)
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":first_stall_beta"
        ] = 15.1735 * (length_width_ratio**-0.618)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        length_height_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":length_height_ratio"
        ]
        length_width_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":length_width_ratio"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":first_stall_alpha",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":length_height_ratio",
        ] = -0.618 * 15.1735 * (length_height_ratio**-1.618)
        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":first_stall_beta",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":length_width_ratio",
        ] = -0.618 * 15.1735 * (length_width_ratio**-1.618)
