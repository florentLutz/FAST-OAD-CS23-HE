# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzleContractionLossCoefficient(om.ExplicitComponent):
    """
    Computation of the pressure contraction loss coefficient of the nozzle.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="nozzle_id",
            default=None,
            desc="Identifier of the nozzle",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_height_length_ratio",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":alpha",
            val=np.nan,
            units="deg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
            units="unitless",
            val=2.0,
        )

        self.add_output(
            "nozzle_contraction_loss_coefficient",
            val=0.3,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        exit_height_length_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_height_length_ratio"
        ]
        alpha = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":alpha"
        ]
        area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio"
        ]

        outputs["nozzle_contraction_loss_coefficient"] = np.where(
            alpha > 30.0,
            (
                0.568
                - 0.01002 * alpha
                - 1.55 * exit_height_length_ratio
                + 0.0001188 * alpha**2.0
                + 0.00276 * alpha * exit_height_length_ratio
                + 1.86 * exit_height_length_ratio**2.0
            )
            * (1.0 - 1.0 / area_ratio),
            (
                0.621
                - 0.01886 * alpha
                - 1.64 * exit_height_length_ratio
                + 0.0003816 * alpha**2.0
                - 0.00804 * alpha * exit_height_length_ratio
                + 2.21 * exit_height_length_ratio**2.0
            )
            * (1.0 - 1.0 / area_ratio),
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        exit_height_length_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_height_length_ratio"
        ]
        alpha = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":alpha"
        ]
        area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio"
        ]

        partials[
            "nozzle_contraction_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_height_length_ratio",
        ] = np.where(
            alpha > 30.0,
            (-1.55 + 0.00276 * alpha + 3.72 * exit_height_length_ratio) * (1.0 - 1.0 / area_ratio),
            (-1.64 - 0.00804 * alpha + 4.41 * exit_height_length_ratio) * (1.0 - 1.0 / area_ratio),
        )
        partials[
            "nozzle_contraction_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":alpha",
        ] = np.where(
            alpha > 30.0,
            (-0.01002 + 0.0002376 * alpha + 0.00276 * exit_height_length_ratio)
            * (1.0 - 1.0 / area_ratio),
            (-0.01886 + 0.0007632 * alpha - 0.00804 * exit_height_length_ratio)
            * (1.0 - 1.0 / area_ratio),
        )
        partials[
            "nozzle_contraction_loss_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
        ] = np.where(
            alpha > 30.0,
            (
                0.568
                - 0.01002 * alpha
                - 1.55 * exit_height_length_ratio
                + 0.0001188 * alpha**2.0
                + 0.00276 * alpha * exit_height_length_ratio
                + 1.86 * exit_height_length_ratio**2.0
            )
            * (1.0 / area_ratio**2.0),
            (
                0.621
                - 0.01886 * alpha
                - 1.64 * exit_height_length_ratio
                + 0.0003816 * alpha**2.0
                - 0.00804 * alpha * exit_height_length_ratio
                + 2.21 * exit_height_length_ratio**2.0
            )
            * (1.0 / area_ratio**2.0),
        )
