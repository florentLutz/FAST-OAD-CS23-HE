# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCSMPMSMOperationalCost(om.ExplicitComponent):
    """
    Computation of the maintenance cost of the SM PMSM. As the bearing accounts most of the
    mechanical faults of the rotor :cite:`orlowska:2022`, the PMSM operational cost is estimated
    based on the bearing's cost and life expectancy. The suggested bearing life expectancy is
    given by Shigley's mechenical engineering design :cite:`shigley:2014`. The bearing types are
    provided from :cite:`emrax:2018`. The off-the-shelf price of the bearing is referenced from
    https://qualitybearingsonline.co.uk/.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max",
            val=np.nan,
            units="kN*m",
            desc="Max continuous torque of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            val=np.nan,
            units="min**-1",
        )
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
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

        torque_max = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max"]
        rpm_rating = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":operational_cost"] = (
            flight_hours_per_year * 0.0311 * (np.pi * torque_max * rpm_rating / 60.0) ** 0.505
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        torque_max = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max"]
        rpm_rating = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = 0.0311 * (np.pi * torque_max * rpm_rating / 60.0) ** 0.505

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":operational_cost",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max",
        ] = (
            0.505
            * flight_hours_per_year
            * 0.0311
            * (np.pi * rpm_rating / 60.0) ** 0.505
            / (torque_max**0.495)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":operational_cost",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
        ] = (
            0.505
            * flight_hours_per_year
            * 0.0311
            * (np.pi * torque_max / 60.0) ** 0.505
            / (rpm_rating**0.495)
        )
