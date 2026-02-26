# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAirMassFlowRatio(om.ExplicitComponent):
    """
    Computation of the inlet air mass flow ratio.
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
            "modified_mass_flow_ratio",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:throat_height",
            val=np.nan,
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:lip_ramp_floor_distance",
            val=np.nan,
            units="m",
        )

        self.add_output(
            "air_mass_flow_ratio",
            val=0.56,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        modified_mass_flow_ratio = inputs["modified_mass_flow_ratio"]
        throat_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:throat_height"
        ]
        lip_ramp_floor_distance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:lip_ramp_floor_distance"
        ]

        outputs["air_mass_flow_ratio"] = (
            modified_mass_flow_ratio * throat_height / lip_ramp_floor_distance
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        modified_mass_flow_ratio = inputs["modified_mass_flow_ratio"]
        throat_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:throat_height"
        ]
        lip_ramp_floor_distance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:lip_ramp_floor_distance"
        ]

        partials["air_mass_flow_ratio", "modified_mass_flow_ratio"] = (
            throat_height / lip_ramp_floor_distance
        )

        partials[
            "air_mass_flow_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:throat_height",
        ] = modified_mass_flow_ratio / lip_ramp_floor_distance

        partials[
            "air_mass_flow_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:lip_ramp_floor_distance",
        ] = -modified_mass_flow_ratio * throat_height / lip_ramp_floor_distance**2.0
