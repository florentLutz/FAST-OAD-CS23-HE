# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesModifiedMassFlowRatio(om.ExplicitComponent):
    """
    Computation of the inlet modified mass flow rate ratio, which is part of the air mass flow rate
    ratio calculation.
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
            + ":air_inlet:throat_height_layer_thickness_ratio",
            val=np.nan,
            units="unitless",
        )

        self.add_output(
            "modified_mass_flow_ratio",
            val=1e-4,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs["modified_mass_flow_ratio"] = (
            0.1651
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_inlet:throat_height_layer_thickness_ratio"
            ]
            ** -0.4068
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        partials[
            "modified_mass_flow_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:throat_height_layer_thickness_ratio",
        ] = (
            -0.06716268
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_inlet:throat_height_layer_thickness_ratio"
            ]
            ** -1.4068
        )
