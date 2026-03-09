# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from sympy.physics.units import volume

from ..fluid_characteristics.fluid_density import FluidDensity


class PerformancesPipeDarcyFrictionFactor(om.ExplicitComponent):
    """
    Maximum pressure drop computation of the humidifier during mission.
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
            + ":material_absolute_roughness",
            units="m",
            val=1.5e-6,
            desc="Absolute roughness of the pipe material, default is the absolute roughness of "
            "copper",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
            units="unitless",
            val=np.nan,
            desc="Reynolds number of the flow in the pipe",
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
            + ":darcy_friction_factor",
            units="unitless",
            val=0.001,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        roughness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":material_absolute_roughness"
        ]
        reynolds_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number"
        ]
        radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius"
        ]

        if reynolds_number <= 2000.0:
            darcy_friction_factor = 64.0 / reynolds_number
        else:
            darcy_friction_factor = (
                1.8 * (-np.log10(roughness / (2.0 * radius) / 3.7)) ** 1.11 + 6.9 / reynolds_number
            ) ** -2.0

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":darcy_friction_factor"
        ] = darcy_friction_factor

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        roughness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":material_absolute_roughness"
        ]
        reynolds_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number"
        ]
        radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius"
        ]

        if reynolds_number <= 2000.0:
            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":darcy_friction_factor",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":reynolds_number",
            ] = -64.0 / reynolds_number**2.0

            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":darcy_friction_factor",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":material_absolute_roughness",
            ] = 0.0

            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":darcy_friction_factor",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":radius",
            ] = 0.0

        else:
            common_term = (
                1.8 * (-np.log10(roughness / (7.4 * radius))) ** 1.11 + 6.9 / reynolds_number
            )

            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":darcy_friction_factor",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":material_absolute_roughness",
            ] = (
                (3.996 / (roughness * np.log(10)))
                * (-np.log10(roughness / (7.4 * radius))) ** 0.11
                * common_term**-3.0
            )

            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":darcy_friction_factor",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":radius",
            ] = (
                -(3.996 / (radius * np.log(10)))
                * (-np.log10(roughness / (7.4 * radius))) ** 0.11
                * common_term**-3.0
            )

            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":darcy_friction_factor",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":reynolds_number",
            ] = 13.8 / reynolds_number**2.0 * common_term**-3.0
