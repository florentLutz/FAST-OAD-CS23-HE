# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPipeReynoldsNumber(om.ExplicitComponent):
    """
    Pipe flow reynolds number computation.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="pipe_id",
            default=None,
            desc="Identifier of the pipe",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":transport_velocity",
            units="m/s",
            val=1.5,
            desc="Pipe flow velocity",
        )
        self.add_input(
            name="coolant_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="coolant_dynamic_viscosity",
            units="Pa*s",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
            units="unitless",
            val=2.7e4,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        flow_velocity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":transport_velocity"
        ]
        fluid_density = inputs["coolant_density"]
        dynamic_viscosity = inputs["coolant_dynamic_viscosity"]
        pipe_radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number"
        ] = 2.0 * pipe_radius * flow_velocity * fluid_density / dynamic_viscosity

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        flow_velocity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":transport_velocity"
        ]
        fluid_density = inputs["coolant_density"]
        dynamic_viscosity = inputs["coolant_dynamic_viscosity"]
        pipe_radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":transport_velocity",
        ] = 2.0 * pipe_radius * fluid_density / dynamic_viscosity

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
            "coolant_density",
        ] = 2.0 * pipe_radius * flow_velocity / dynamic_viscosity

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
            "coolant_dynamic_viscosity",
        ] = -2.0 * pipe_radius * flow_velocity * fluid_density / dynamic_viscosity**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius",
        ] = 2.0 * flow_velocity * fluid_density / dynamic_viscosity
