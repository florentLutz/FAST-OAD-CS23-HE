# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingTotalTransferAreaVolumeRatio(om.ExplicitComponent):
    """
    Computation of the fraction of total heat transfer area and total volume for both side of the
    heat exchange system.
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
            + ":heat_exchanger:surface_area_density",
            units="1/m",
            val=2254.0,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thickness",
            units="m",
            val=8e-4,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio",
            units="1/m",
            val=1.0,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        plate_spacing = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_spacing"
        ]
        surface_area_density = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:surface_area_density"
        ]
        plate_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thickness"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio"
        ] = surface_area_density * plate_spacing / (2.0 * (plate_thickness + plate_spacing))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        plate_spacing = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_spacing"
        ]
        surface_area_density = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:surface_area_density"
        ]
        plate_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thickness"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:surface_area_density",
        ] = plate_spacing / (2.0 * (plate_thickness + plate_spacing))

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_spacing",
        ] = (
            surface_area_density
            * plate_thickness
            / (2.0 * (plate_thickness + plate_spacing) ** 2.0)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thickness",
        ] = -surface_area_density * plate_spacing / (2.0 * (plate_thickness + plate_spacing) ** 2.0)
