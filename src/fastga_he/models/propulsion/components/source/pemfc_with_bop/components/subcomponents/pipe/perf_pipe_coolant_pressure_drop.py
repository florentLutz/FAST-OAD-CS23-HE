# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPipeCoolantPressureDrop(om.ExplicitComponent):
    """
    Maximum pressure drop computation of the connection pipe during mission.
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
            name="coolant_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:transport_velocity",
            units="m/s",
            val=1.5,
            desc="Pipe flow velocity",
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
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":darcy_friction_factor",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":number_of_pipes",
            units="unitless",
            val=np.nan,
            desc="Number of the coolant pipes in the TMS",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":coolant_pressure_drop",
            units="Pa",
            val=1e3,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        density = inputs["coolant_density"]
        velocity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:transport_velocity"
        ]
        radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius"
        ]
        darcy_friction_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":darcy_friction_factor"
        ]
        number_of_pipes = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":number_of_pipes"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":coolant_pressure_drop"
        ] = darcy_friction_factor * 0.5 * number_of_pipes * density * velocity**2.0 / (4.0 * radius)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        density = inputs["coolant_density"]
        velocity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:transport_velocity"
        ]
        radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius"
        ]
        darcy_friction_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":darcy_friction_factor"
        ]
        number_of_pipes = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":number_of_pipes"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":coolant_pressure_drop",
            "coolant_density",
        ] = darcy_friction_factor * 0.5 * number_of_pipes * velocity**2.0 / (4.0 * radius)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":coolant_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:transport_velocity",
        ] = darcy_friction_factor * density * velocity * number_of_pipes / (4.0 * radius)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":coolant_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius",
        ] = (
            -darcy_friction_factor
            * 0.5
            * number_of_pipes
            * density
            * velocity**2.0
            / (4.0 * radius**2.0)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":coolant_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":darcy_friction_factor",
        ] = 0.5 * number_of_pipes * density * velocity**2.0 / (4.0 * radius)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":coolant_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":number_of_pipes",
        ] = darcy_friction_factor * 0.5 * density * velocity**2.0 / (4.0 * radius)
