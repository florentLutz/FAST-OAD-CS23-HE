# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingPipeWallThickness(om.ExplicitComponent):
    """
    The thickness of the pipe, which is calculated from the hoop stress formula.
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
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:static_pressure",
            units="Pa",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:radius",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:safety_factor",
            units="unitless",
            val=1.5,
            desc="Safety factor for the pipe wall thickness, default is 1.5",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:material_yield_strength",
            units="Pa",
            val=40e6,
            desc="Yield strength of the pipe material, default is 40.0 MPa for copper",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:wall_thickness",
            units="m",
            val=0.005,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":pipe:radius"
        ]
        safety_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:safety_factor"
        ]
        yield_strength = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:material_yield_strength"
        ]
        static_pressure = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:static_pressure"
        ]

        if yield_strength <= static_pressure:
            raise ValueError(
                f"The static pressure ({static_pressure:.2f} Pa) exceeds the yield strength "
                f"({yield_strength:.2f} Pa) of the pipe material. "
                "The pipe will fail under these conditions. Please check the inputs."
            )

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:wall_thickness"
        ] = (static_pressure * radius * safety_factor) / (yield_strength - static_pressure)

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":pipe:radius"
        ]
        safety_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:safety_factor"
        ]
        yield_strength = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:material_yield_strength"
        ]
        static_pressure = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:static_pressure"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:wall_thickness",
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":pipe:radius",
        ] = (static_pressure * safety_factor) / (yield_strength - static_pressure)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:wall_thickness",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:safety_factor",
        ] = (static_pressure * radius) / (yield_strength - static_pressure)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:wall_thickness",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:material_yield_strength",
        ] = -(static_pressure * radius * safety_factor) / (yield_strength - static_pressure) ** 2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":pipe:wall_thickness",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:static_pressure",
        ] = (radius * safety_factor * yield_strength) / (yield_strength - static_pressure) ** 2.0
