# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCSplitterDimensions(om.ExplicitComponent):
    """
    Computation of the splitter dimensions, assumes two conducting plates surrounded by an
    insulation layer whose thickness is computed based on PD criterion.
    """

    def initialize(self):

        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            types=str,
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
            name="data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":insulation:thickness",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":height",
            units="m",
            val=10e-3,
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":width",
            units="m",
            val=20e-3,
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":length",
            units="m",
            val=0.3,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":height",
            wrt=[
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":cross_section:thickness",
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":insulation:thickness",
            ],
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":width",
            wrt=[
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":cross_section:width",
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":insulation:thickness",
            ],
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":length",
            wrt=[
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":cross_section:length",
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":insulation:thickness",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]

        # One plate conducting and one plate for return of current
        outputs["data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":height"] = (
            2.0
            * inputs[
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":cross_section:thickness"
            ]
            + 3.0
            * inputs[
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":insulation:thickness"
            ]
        )

        outputs["data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":width"] = (
            inputs[
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":cross_section:width"
            ]
            + 2.0
            * inputs[
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":insulation:thickness"
            ]
        )

        outputs["data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":length"] = (
            inputs[
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":cross_section:length"
            ]
            + 2.0
            * inputs[
                "data:propulsion:he_power_train:DC_splitter:"
                + dc_splitter_id
                + ":insulation:thickness"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]

        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":height",
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:thickness",
        ] = 2.0
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":height",
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":insulation:thickness",
        ] = 3.0

        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":width",
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":cross_section:width",
        ] = 1.0
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":width",
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":insulation:thickness",
        ] = 2.0

        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":length",
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:length",
        ] = 1.0
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":length",
            "data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":insulation:thickness",
        ] = 2.0
