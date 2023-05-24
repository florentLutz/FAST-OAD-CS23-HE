# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingRectifierContactorWeight(om.ExplicitComponent):
    """
    Computation of the weight of the contactors, based on a regression performed on the data from
    :cite:`giraud:2014`. Assumes that there are 3 contactors in the rectifier. Correlation can be
    found in ..methodology.contactor_mass.py.
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
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":contactor:mass",
            units="kg",
            val=4.5,
            desc="Mass of the 3 contactors",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        rectifier_id = self.options["rectifier_id"]

        current_ac_caliber = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":contactor:mass"] = (
            3.0 * 0.0239 * current_ac_caliber ** 0.6942
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        current_ac_caliber = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":contactor:mass",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        ] = (
            3.0 * 0.0239 * 0.6942 * current_ac_caliber ** (0.6942 - 1.0)
        )
