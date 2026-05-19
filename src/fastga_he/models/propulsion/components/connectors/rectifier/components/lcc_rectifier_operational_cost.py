# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCRectifierOperationalCost(om.ExplicitComponent):
    """
    Computation of the annual operational cost of rectifier. The diodes are considered as the
    throttle components in the system. The estimated rectifier life expectancy is based on the
    lifespan of the IGBTs given by :cite:`sathik:2018` .
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
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":lifespan",
            units="h",
            val=3.4e4,
            desc="Expected lifetime of the rectifier, based on the lifespan of the IGBTs",
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            units="USD/yr",
            val=28.7,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        purchase_cost = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost"
        ]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]
        lifespan = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":lifespan"]

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost"
        ] = purchase_cost * flight_hours_per_year / lifespan

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        purchase_cost = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost"
        ]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]
        lifespan = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":lifespan"]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost",
        ] = flight_hours_per_year / lifespan

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = purchase_cost / lifespan

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":lifespan",
        ] = -purchase_cost * flight_hours_per_year / lifespan**2.0
