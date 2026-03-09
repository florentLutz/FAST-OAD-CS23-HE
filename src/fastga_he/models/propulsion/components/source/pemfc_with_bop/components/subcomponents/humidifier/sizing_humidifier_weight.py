# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHumidifierWeight(om.ExplicitComponent):
    """
    Weight computation of the humidifier.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="humidifier_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        humidifier_id = self.options["humidifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":power_rating",
            units="kW",
            val=np.nan,
        )

        ## Output
        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + humidifier_id
            + ":mass",
            units="kg",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        humidifier_id = self.options["humidifier_id"]

        pemfc_power_rating = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_rating"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + humidifier_id
            + ":mass"
        ] = np.where(
            pemfc_power_rating <= 200,
            4.535e-5 * pemfc_power_rating**2.0 + 0.062 * pemfc_power_rating + 2.119,
            16.33 * pemfc_power_rating / 200,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        humidifier_id = self.options["humidifier_id"]

        pemfc_power_rating = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_rating"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + humidifier_id
            + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":power_rating",
        ] = np.where(
            pemfc_power_rating <= 200,
            9.07e-05 * pemfc_power_rating + 0.062,
            16.33 / 200,
        )
