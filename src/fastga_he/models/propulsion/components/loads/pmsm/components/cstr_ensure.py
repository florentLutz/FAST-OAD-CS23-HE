# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_PMSM

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PMSM, "fastga_he.submodel.propulsion.constraints.pmsm.ensure"
)
class ConstraintsEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum seen by the pmsm during the
    mission and the value used for sizing, ensuring each component works below its maxima.
    """

    def initialize(self):

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_max",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the torque the motor has to provide",
        )
        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating",
            units="N*m",
            val=np.nan,
            desc="Max continuous torque of the motor",
        )
        self.add_output(
            "constraints:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating",
            units="N*m",
            val=0.0,
            desc="Respected if <0",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating",
            wrt=[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_max",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating",
            ],
            method="exact",
        )

        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_max",
            units="min**-1",
            val=np.nan,
            desc="Maximum value of the motor rpm during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            units="min**-1",
            val=np.nan,
            desc="Max continuous rpm of the motor",
        )
        self.add_output(
            "constraints:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            units="min**-1",
            val=0.0,
            desc="Respected if <0",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            wrt=[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_max",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        outputs["constraints:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating"] = (
            inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_max"]
            - inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating"]
        )
        outputs["constraints:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating"] = (
            inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_max"]
            - inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        partials[
            "constraints:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating",
        ] = -1.0

        partials[
            "constraints:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
        ] = -1.0
