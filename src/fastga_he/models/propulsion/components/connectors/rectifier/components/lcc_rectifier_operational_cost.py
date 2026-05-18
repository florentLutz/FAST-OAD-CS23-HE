# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

EXPECTED_LIFESPAN = 1.26e4  # Expected lifespan of the diode in hours, with a safety factor of 1.5


class LCCRectifierOperationalCost(om.ExplicitComponent):
    """
    Computation of the annual operational cost of rectifier. The diodes are considered as the
    throttle components in the system. Thw estimated rectifier life expectancy is based on the
    lifespan of the IGBTs given by :cite:`sathik:2018` and a reduction factor of 0.3 based on the
    result from :cite:`infineon:2021`.
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

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            units="USD/yr",
            val=350.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost"
        ] = (
            inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost"]
            * inputs["data:TLAR:flight_hours_per_year"]
            / EXPECTED_LIFESPAN
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost",
        ] = inputs["data:TLAR:flight_hours_per_year"] / EXPECTED_LIFESPAN

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = (
            inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":purchase_cost"]
            / EXPECTED_LIFESPAN
        )
