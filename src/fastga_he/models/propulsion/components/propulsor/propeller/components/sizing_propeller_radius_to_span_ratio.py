# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingPropellerDiameterToSpanRatio(om.ExplicitComponent):
    """
    Computation of the wing chord behind the wing.
    """

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output(
            name="data:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":diameter_to_span_ratio",
            val=0.1,
            desc="Diameter of the propeller as a ratio of the wing half span",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        outputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio"
        ] = (
            2.0
            * inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
            / inputs["data:geometry:wing:span"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = (
            2.0 / inputs["data:geometry:wing:span"]
        )
        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio",
            "data:geometry:wing:span",
        ] = -(
            2.0
            * inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
            / inputs["data:geometry:wing:span"] ** 2.0
        )
