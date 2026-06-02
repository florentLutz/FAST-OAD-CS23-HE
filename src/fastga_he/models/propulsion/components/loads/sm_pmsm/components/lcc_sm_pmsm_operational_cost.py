# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCSMPMSMOperationalCost(om.ExplicitComponent):
    """
    Computation of the maintenance cost of the SM PMSM. As the bearing accounts for most of the
    mechanical faults of the rotor :cite:`orlowska:2022`, the PMSM lifespan is estimated
    based on the bearing life expectancy, given by Shigley's mechanical engineering design
    :cite:`shigley:2014`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":purchase_cost",
            units="USD",
            val=np.nan,
            desc="Unit purchase cost of the PMS motor",
        )
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":lifespan",
            val=2000.0,
            units="h",
            desc="Expected lifespan of the PMSM motor, based on the bearing life expectancy",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":operational_cost",
            units="USD/yr",
            val=1.0e3,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        purchase_cost = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":purchase_cost"
        ]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]
        lifespan = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":lifespan"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":operational_cost"] = (
            purchase_cost * flight_hours_per_year / lifespan
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        purchase_cost = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":purchase_cost"
        ]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]
        lifespan = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":lifespan"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = purchase_cost / lifespan

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":operational_cost",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":lifespan",
        ] = -purchase_cost * flight_hours_per_year / lifespan**2.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":operational_cost",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":purchase_cost",
        ] = flight_hours_per_year / lifespan
