# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingPropellerInputWeight(om.ExplicitComponent):
    """
    Quickly adds a way to have the propeller weight as an input rather than be computed.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":input_mass",
            val=np.nan,
            units="kg",
            desc="Propeller mass as an input, because of some convention adopted, it must be a different variable than the actual weight",
        )
        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":mass",
            val=20.0,
            units="kg",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":mass"] = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":input_mass"
        ]
