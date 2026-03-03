# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHEXFinGeometry(om.ExplicitComponent):
    """
    Fin geometry sizing for heat exchanger.
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
            + ":heat_exchanger:fin_spacing",
            units="m",
            val=1.18e-3,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness",
            units="m",
            val=1.02e-4,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height",
            units="m",
            val=6.25e-3,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_length",
            units="m",
            val=3.175e-3,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing_height_ratio",
            units="unitless",
            val=0.01,
        )
        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness_length_ratio",
            units="unitless",
            val=0.01,
        )
        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness_spacing_ratio",
            units="unitless",
            val=0.01,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:fin_thickness_length_ratio",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:fin_thickness_spacing_ratio",
            ],
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness",
            method="exact",
        )
        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness_length_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_length",
            method="exact",
        )
        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing_height_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height",
            method="exact",
        )
        self.declare_partials(
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:fin_spacing_height_ratio",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:fin_thickness_spacing_ratio",
            ],
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        fin_spacing = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing"
        ]
        fin_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness"
        ]
        fin_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height"
        ]
        fin_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_length"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing_height_ratio"
        ] = fin_spacing / fin_height
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness_length_ratio"
        ] = fin_thickness / fin_length
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness_spacing_ratio"
        ] = fin_thickness / fin_spacing

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        fin_spacing = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing"
        ]
        fin_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness"
        ]
        fin_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height"
        ]
        fin_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_length"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness_length_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness",
        ] = 1.0 / fin_length

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness_length_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_length",
        ] = -fin_thickness / fin_length**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing_height_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height",
        ] = -fin_spacing / fin_height**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing_height_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing",
        ] = 1.0 / fin_height

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness_spacing_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness",
        ] = 1.0 / fin_spacing

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness_spacing_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing",
        ] = -fin_thickness / fin_spacing**2.0
