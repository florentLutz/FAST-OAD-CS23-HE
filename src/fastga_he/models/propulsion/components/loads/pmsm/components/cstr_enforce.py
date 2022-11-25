# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_PMSM

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_PMSM
] = "fastga_he.submodel.propulsion.constraints.pmsm.enforce"


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PMSM, "fastga_he.submodel.propulsion.constraints.pmsm.enforce"
)
class ConstraintsEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maxima seen by the motor during the mission are used for the
    sizing, ensuring a fitted design of each component.
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
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_max",
            units="min**-1",
            val=np.nan,
            desc="Maximum value of the motor rpm during the mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating",
            units="N*m",
            val=250.0,
            desc="Max continuous torque of the motor",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_max",
            val=1.0,
        )

        self.add_output(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            units="min**-1",
            val=5000.0,
            desc="Max continuous rpm of the motor",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating"] = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_max"
        ]
        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating"] = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_max"
        ]
