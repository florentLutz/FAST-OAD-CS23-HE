# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCDZeroInletMassFlow(om.ExplicitComponent):
    """
    Computation of the inlet drag coefficient with zero inlet mass flow rate.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="air_inlet_id",
            default=None,
            desc="Identifier of the air inlet",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lip_length",
            val=np.nan,
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_floor_inlet_plane_distance",
            val=np.nan,
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lip_ramp_floor_distance",
            val=np.nan,
            units="m",
        )

        self.add_output(
            "cd_zero_inlet_mass_flow",
            val=0.16,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        lip_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lip_length"
        ]
        ramp_floor_inlet_plane_distance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_floor_inlet_plane_distance"
        ]
        lip_ramp_floor_distance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lip_ramp_floor_distance"
        ]

        outputs["cd_zero_inlet_mass_flow"] = (
            0.1362
            * ((ramp_floor_inlet_plane_distance - lip_ramp_floor_distance) / lip_length) ** -0.2202
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        lip_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lip_length"
        ]
        ramp_floor_inlet_plane_distance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_floor_inlet_plane_distance"
        ]
        lip_ramp_floor_distance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lip_ramp_floor_distance"
        ]

        partials[
            "cd_zero_inlet_mass_flow",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lip_length",
        ] = (
            0.02999124
            * ((ramp_floor_inlet_plane_distance - lip_ramp_floor_distance) / lip_length) ** -1.2202
            * (ramp_floor_inlet_plane_distance - lip_ramp_floor_distance)
            / lip_length**2.0
        )

        partials[
            "cd_zero_inlet_mass_flow",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_floor_inlet_plane_distance",
        ] = (
            -0.02999124
            * ((ramp_floor_inlet_plane_distance - lip_ramp_floor_distance) / lip_length) ** -1.2202
            / lip_length
        )

        partials[
            "cd_zero_inlet_mass_flow",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lip_ramp_floor_distance",
        ] = (
            0.02999124
            * ((ramp_floor_inlet_plane_distance - lip_ramp_floor_distance) / lip_length) ** -1.2202
            / lip_length
        )
