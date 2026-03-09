# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHeatExchangerFinHydraulicDiameter(om.ExplicitComponent):
    """
    Fin hydraulic diameter of the heat exchanger.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_spacing",
            units="m",
            val=1.18e-3,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_thickness",
            units="m",
            val=1.02e-4,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_height",
            units="m",
            val=6.25e-3,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_length",
            units="m",
            val=3.175e-3,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
            units="m",
            val=2.38e-3,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        fin_spacing = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_spacing"
        ]
        fin_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_thickness"
        ]
        fin_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_height"
        ]
        fin_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_length"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter"
        ] = (
            4.0
            * fin_spacing
            * fin_height
            * fin_length
            / (
                2.0
                * (fin_spacing * fin_length + fin_height * fin_length + fin_thickness * fin_height)
                + fin_spacing * fin_thickness
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        fin_spacing = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_spacing"
        ]
        fin_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_thickness"
        ]
        fin_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_height"
        ]
        fin_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_length"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_spacing",
        ] = (8.0 * fin_height**2.0 * fin_length * (fin_thickness + fin_length)) / (
            (fin_thickness + 2.0 * fin_length) * fin_spacing
            + 2.0 * fin_height * fin_thickness
            + 2.0 * fin_height * fin_length
        ) ** 2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_thickness",
        ] = (
            -(4.0 * fin_height * fin_length * fin_spacing * (fin_spacing + 2.0 * fin_height))
            / (
                2.0
                * (fin_height * fin_thickness + fin_length * fin_spacing + fin_height * fin_length)
                + fin_spacing * fin_thickness
            )
            ** 2.0
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_height",
        ] = (4.0 * fin_length * fin_spacing**2.0 * (fin_thickness + 2.0 * fin_length)) / (
            (2.0 * fin_thickness + 2.0 * fin_length) * fin_height
            + fin_spacing * fin_thickness
            + 2.0 * fin_length * fin_spacing
        ) ** 2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_length",
        ] = (4.0 * fin_height * fin_spacing * (fin_spacing + 2.0 * fin_height) * fin_thickness) / (
            (2.0 * fin_spacing + 2.0 * fin_height) * fin_length
            + (fin_spacing + 2.0 * fin_height) * fin_thickness
        ) ** 2.0
