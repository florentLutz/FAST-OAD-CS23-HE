# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingInletWeight(om.ExplicitComponent):
    """
    Computation of the inlet weight.
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
            + ":length",
            val=np.nan,
            units="ft",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":inlet_capture_area",
            val=np.nan,
            units="ft**2",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_ambient_dynamic_pressure",
            val=np.nan,
            units="psi",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass",
            val=10.0,
            units="lb",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":length"
        ]
        inlet_capture_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":inlet_capture_area"
        ]
        design_ambient_dynamic_pressure = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_ambient_dynamic_pressure"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass"
        ] = (
            0.32 * length * inlet_capture_area**0.65 * design_ambient_dynamic_pressure**0.6
            + 1.735
            * (length * np.sqrt(inlet_capture_area) * design_ambient_dynamic_pressure * 1.3)
            ** 0.7331
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":length"
        ]
        inlet_capture_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":inlet_capture_area"
        ]
        design_ambient_dynamic_pressure = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_ambient_dynamic_pressure"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":length",
        ] = (
            0.32 * inlet_capture_area**0.65 * design_ambient_dynamic_pressure**0.6
            + 1.735
            * 0.7331
            * (length * np.sqrt(inlet_capture_area) * design_ambient_dynamic_pressure * 1.3)
            ** -0.2669
            * np.sqrt(inlet_capture_area)
            * design_ambient_dynamic_pressure
            * 1.3
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":inlet_capture_area",
        ] = (
            0.32 * length * 0.65 * inlet_capture_area**-0.35 * design_ambient_dynamic_pressure**0.6
            + 1.735
            * 0.7331
            * (length * np.sqrt(inlet_capture_area) * design_ambient_dynamic_pressure * 1.3)
            ** -0.2669
            * length
            * 0.5
            / np.sqrt(inlet_capture_area)
            * design_ambient_dynamic_pressure
            * 1.3
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_ambient_dynamic_pressure",
        ] = (
            0.192 * length * inlet_capture_area**0.65 * design_ambient_dynamic_pressure**-0.4
            + 1.2719285
            * (length * np.sqrt(inlet_capture_area) * 1.3) ** 0.7331
            * design_ambient_dynamic_pressure**-0.2669
        )
