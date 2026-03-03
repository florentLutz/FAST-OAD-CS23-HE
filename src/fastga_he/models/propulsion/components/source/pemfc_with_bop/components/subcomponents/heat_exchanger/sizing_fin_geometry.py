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
            + ":heat_exchanger:plate_spacing",
            units="m",
            val=6.35e-3,
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
            + ":heat_exchanger:fin_frequency",
            units="1/m",
            val=782,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height",
            units="m",
            val=6.25e-3,
        )
        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing",
            units="m",
            val=1.18e-3,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness",
            val=-1.0,
        )
        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_spacing",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_frequency",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        plate_spacing = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_spacing"
        ]
        fin_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness"
        ]
        fin_frequency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_frequency"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height"
        ] = plate_spacing - fin_thickness
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing"
        ] = 1.0 / fin_frequency - fin_thickness

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_spacing",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_frequency",
        ] = (
            -1.0
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:fin_frequency"
            ]
            ** 2.0
        )
