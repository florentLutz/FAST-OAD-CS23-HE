# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPressureDropCoefficient(om.ExplicitComponent):
    """
    Computation of the pressure drop coefficient for entrance and exit flow of the heat exchanger.
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
            + ":heat_exchanger:free_flow_frontal_area_ratio",
            units="unitless",
            val=np.nan,
        )

        self.add_output(
            name="entrance_pressure_drop_coefficient",
            units="unitless",
            val=2.64,
        )
        self.add_output(
            name="exit_pressure_drop_coefficient",
            units="unitless",
            val=1.0,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        sigma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:free_flow_frontal_area_ratio"
        ]

        outputs["entrance_pressure_drop_coefficient"] = 0.5 * (1.0 - sigma) - 1.0 + sigma**2.0
        outputs["exit_pressure_drop_coefficient"] = (1.0 - sigma) ** 2.0 + 1.0 - sigma**2.0

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        sigma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:free_flow_frontal_area_ratio"
        ]

        partials[
            "entrance_pressure_drop_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:free_flow_frontal_area_ratio",
        ] = -0.5 + 2.0 * sigma

        partials[
            "exit_pressure_drop_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:free_flow_frontal_area_ratio",
        ] = -2.0
