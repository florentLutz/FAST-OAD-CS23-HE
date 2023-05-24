# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingRectifierInductorCurrentCaliber(om.ExplicitComponent):
    """
    Computation of the maximum current that can pass through the inductor. It is gonna be equal
    to the current caliber in one arm of the rectifier but we need this component to change the
    name
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
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":inductor:current_caliber",
            units="A",
            val=200.0,
            desc="Current caliber of one of the inductor of the rectifier",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":inductor:current_caliber",
            wrt="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        rectifier_id = self.options["rectifier_id"]
        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":inductor:current_caliber"
        ] = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]
