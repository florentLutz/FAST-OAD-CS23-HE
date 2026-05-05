# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesPEMFCStackBOPOxidizerTemperature(om.ExplicitComponent):
    """
    Oxidizer temperature computation of the PEMFC stack, using for BOP and TMS sizing.
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
            + ":coolant:inlet_temperature",
            units="K",
            val=np.nan,
            desc="standard operating temperature for the PEMFC",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":oxidizer_temperature",
            units="K",
            val=320.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":oxidizer_temperature"
        ] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":coolant:inlet_temperature"
            ]
            + 1.0
        )
