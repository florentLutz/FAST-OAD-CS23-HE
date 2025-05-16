# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCRectifierOperationalCost(om.ExplicitComponent):
    """
    Computation of the annual operational cost of rectifier.
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
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost",
            units="USD",
            val=np.nan,
            desc="Maximum RMS current flowing through one arm of the rectifier",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":lifespan",
            units="yr",
            val=15.0,
            desc="Expected lifetime of the rectifier, typically around 15 year",
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            units="USD/yr",
            val=350.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]
        cost = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost"]
        lifespan = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":lifespan"]

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost"
        ] = cost / lifespan

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]
        cost = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost"]
        lifespan = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":lifespan"]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost",
        ] = 1.0 / lifespan

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":lifespan",
        ] = -cost / lifespan**2.0
