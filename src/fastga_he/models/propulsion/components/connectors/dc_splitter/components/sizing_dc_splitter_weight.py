# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingDCSplitterWeight(om.ExplicitComponent):
    """
    Computation of the weight of a DC splitter.
    """

    def initialize(self):
        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            allow_none=False,
        )

    def setup(self):

        dc_splitter_id = self.options["dc_splitter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:thickness",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:width",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:length",
            units="m",
            val=0.3,
            desc="Length of the splitter conductor",
        )

        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":height",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":width",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:density",
            val=8960.0,
            units="kg/m**3",
            desc="Density of the conductor, copper is assumed",
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_splitter:insulation:density",
            units="kg/m**3",
            val=1450.0,
            desc="Density of the insulation, Gexol is assumed",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            units="kg",
            val=1.0,
            desc="Weight of the splitter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_splitter_id = self.options["dc_splitter_id"]

        conductor_thickness = inputs[
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:thickness"
        ]
        conductor_width = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":cross_section:width"
        ]
        conductor_length = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":cross_section:length"
        ]
        conductor_density = inputs[
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:density"
        ]

        splitter_height = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":height"
        ]
        splitter_width = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":width"
        ]
        splitter_length = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":length"
        ]
        insulation_density = inputs[
            "settings:propulsion:he_power_train:DC_splitter:insulation:density"
        ]

        conductor_volume = 2.0 * conductor_thickness * conductor_width * conductor_length
        insulation_volume = splitter_height * splitter_width * splitter_length - conductor_volume

        outputs["data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass"] = (
            conductor_volume * conductor_density + insulation_volume * insulation_density
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_splitter_id = self.options["dc_splitter_id"]

        conductor_thickness = inputs[
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:thickness"
        ]
        conductor_width = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":cross_section:width"
        ]
        conductor_length = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":cross_section:length"
        ]
        conductor_density = inputs[
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:density"
        ]

        splitter_height = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":height"
        ]
        splitter_width = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":width"
        ]
        splitter_length = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":length"
        ]
        insulation_density = inputs[
            "settings:propulsion:he_power_train:DC_splitter:insulation:density"
        ]

        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:thickness",
        ] = (
            2.0 * conductor_width * conductor_length * (conductor_density - insulation_density)
        )
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":cross_section:width",
        ] = (
            2.0 * conductor_thickness * conductor_length * (conductor_density - insulation_density)
        )
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:length",
        ] = (
            2.0 * conductor_width * conductor_thickness * (conductor_density - insulation_density)
        )
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:density",
        ] = (
            2.0 * conductor_width * conductor_thickness * conductor_length
        )

        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":height",
        ] = (
            splitter_width * splitter_length * insulation_density
        )
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":width",
        ] = (
            splitter_height * splitter_length * insulation_density
        )
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":length",
        ] = (
            splitter_width * splitter_height * insulation_density
        )
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            "settings:propulsion:he_power_train:DC_splitter:insulation:density",
        ] = (
            splitter_width * splitter_height * splitter_length
            - 2.0 * conductor_length * conductor_width * conductor_thickness
        )
