# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingRectifierModuleDimension(om.ExplicitComponent):
    """
    Computation of the dimension of one casing. Based on a regression on the SEMIKRON family from
    :cite:`giraud:2014`.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of one arm of the rectifier",
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:length",
            units="m",
            val=0.150,
            desc="Length of one module",
        )
        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:width",
            units="m",
            val=0.065,
            desc="Width of one module",
        )
        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:height",
            units="m",
            val=0.021,
            desc="Height of one module",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:length",
            wrt="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        current_ac_caliber = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:length"] = (
            1.73e-1 * current_ac_caliber + 81
        ) * 1e-3
        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:height"] = (
            0.021
        )
        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:width"] = (
            0.065
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:length",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        ] = 1.73e-4
