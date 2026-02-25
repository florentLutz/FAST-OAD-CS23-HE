# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMomentumFlowCorrectionFactor(om.ExplicitComponent):
    """
    Computation of the momentum flow correction factor of a flush inlet. This is part of the inlet
    ram drag computation.
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
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:layer_thickness_highlight_height_ratio",
            val=np.nan,
            units="unitless",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:momentum_flow_correction_factor",
            val=1e-4,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach"
        ]
        layer_thickness_highlight_height_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:layer_thickness_highlight_height_ratio"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:momentum_flow_correction_factor"
        ] = (
            10 ** (-0.12877)
            * (
                layer_thickness_highlight_height_ratio
                ** (
                    -0.2636
                    - 0.03841 * np.log10(design_mach)
                    + 0.06416 * (np.log10(layer_thickness_highlight_height_ratio)) ** 2.0
                    - 0.11447 * np.log10(layer_thickness_highlight_height_ratio)
                )
            )
            * (design_mach ** (-0.06691 - 0.0682 * np.log10(design_mach)))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach"
        ]
        layer_thickness_highlight_height_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:layer_thickness_highlight_height_ratio"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:momentum_flow_correction_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach",
        ] = (
            (-0.06691 - 0.0682 * np.log10(design_mach))
            * (
                10 ** (-0.12877)
                * (
                    layer_thickness_highlight_height_ratio
                    ** (
                        -0.2636
                        - 0.03841 * np.log10(design_mach)
                        + 0.06416 * (np.log10(layer_thickness_highlight_height_ratio)) ** 2.0
                        - 0.11447 * np.log10(layer_thickness_highlight_height_ratio)
                    )
                )
                * (design_mach ** (-0.06691 - 0.0682 * np.log10(design_mach)))
            )
            / design_mach
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:momentum_flow_correction_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:layer_thickness_highlight_height_ratio",
        ] = (
            (
                -0.2636
                - 0.03841 * np.log10(design_mach)
                + 0.06416 * (np.log10(layer_thickness_highlight_height_ratio)) ** 2.0
                - 0.11447 * np.log10(layer_thickness_highlight_height_ratio)
            )
            * (
                10 ** (-0.12877)
                * (
                    layer_thickness_highlight_height_ratio
                    ** (
                        -0.2636
                        - 0.03841 * np.log10(design_mach)
                        + 0.06416 * (np.log10(layer_thickness_highlight_height_ratio)) ** 2.0
                        - 0.11447 * np.log10(layer_thickness_highlight_height_ratio)
                    )
                )
                * (design_mach ** (-0.06691 - 0.0682 * np.log10(design_mach)))
            )
            / layer_thickness_highlight_height_ratio
        )
