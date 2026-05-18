# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

EXPECTED_LIFESPAN = 4.2e4  # Expected lifespan of the IGBTs in hours, with a safety factor of 1.5


class LCCInverterOperationalCost(om.ExplicitComponent):
    """
    Computation of the inverter annual operational cost. This is estimated based on the lifespan of
    the IGBTs given by :cite:`sathik:2018`, which is indicated as the throttling component.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):
        inverter_id = self.options["inverter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost",
            units="USD",
            val=np.nan,
        )
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost",
            units="USD/yr",
            val=350.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost"] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost"]
            * inputs["data:TLAR:flight_hours_per_year"]
            / EXPECTED_LIFESPAN
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost",
        ] = inputs["data:TLAR:flight_hours_per_year"] / EXPECTED_LIFESPAN

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":purchase_cost"]
            / EXPECTED_LIFESPAN
        )
