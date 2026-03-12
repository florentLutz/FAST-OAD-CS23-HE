# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingEntryHydraulicDiameter(om.ExplicitComponent):
    """
    Computation of the entry hydraulic diameter.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="diffuser_id",
            default=None,
            desc="Identifier of the diffuser",
            allow_none=False,
        )
        self.options.declare(
            name="connected_air_inlet_id",
            default=None,
            desc="Identifier of the connected air inlet",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]
        air_inlet_id = self.options["connected_air_inlet_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height",
            val=np.nan,
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":highlight_width",
            val=np.nan,
            units="m",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":entry_hydraulic_diameter",
            val=0.1,
            units="m",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]
        air_inlet_id = self.options["connected_air_inlet_id"]

        air_inlet_throat_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height"
        ]
        air_inlet_highlight_width = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":highlight_width"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":entry_hydraulic_diameter"
        ] = (
            2.0
            * air_inlet_throat_height
            * air_inlet_highlight_width
            / (air_inlet_throat_height + air_inlet_highlight_width)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]
        air_inlet_id = self.options["connected_air_inlet_id"]

        air_inlet_throat_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height"
        ]
        air_inlet_highlight_width = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":highlight_width"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":entry_hydraulic_diameter",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height",
        ] = (
            air_inlet_highlight_width**2.0
            / (air_inlet_throat_height + air_inlet_highlight_width) ** 2.0
        )
        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":entry_hydraulic_diameter",
            "data:propulsion:he_power_train:PEMFC_stack_bOP:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":highlight_width",
        ] = (
            air_inlet_throat_height**2.0
            / (air_inlet_throat_height + air_inlet_highlight_width) ** 2.0
        )
