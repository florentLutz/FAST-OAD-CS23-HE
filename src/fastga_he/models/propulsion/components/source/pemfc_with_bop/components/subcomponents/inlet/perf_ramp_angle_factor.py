# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesRampAngleFactor(om.ExplicitComponent):
    """
    Computation of the effectiveness of the inlet ramp angle to the inlet drag coefficient.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:ramp_angle",
            val=7.0,
            units="deg",
            desc="Ramp angle of the inlet, defined as the angle between the inlet walls and the "
            "horizontal plane",
        )

        self.add_output(
            "air_mass_flow_ratio",
            val=1e-4,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        ramp_angle = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:ramp_angle"
        ]

        outputs["ramp_angle_factor"] = np.where(
            ramp_angle > 7.0, -0.0121 * ramp_angle**2.0 + 0.3262 * ramp_angle - 0.7183, 1.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        ramp_angle = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:ramp_angle"
        ]

        partials[
            "ramp_angle_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:ramp_angle",
        ] = np.where(ramp_angle > 7.0, -0.0242 * ramp_angle + 0.3262, 1e-6)
