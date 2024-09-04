# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_PMSM_TORQUE,
    SUBMODEL_CONSTRAINTS_PMSM_RPM,
    SUBMODEL_CONSTRAINTS_PMSM_VOLTAGE,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_PMSM_TORQUE] = (
    "fastga_he.submodel.propulsion.constraints.pmsm.torque.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_PMSM_RPM] = (
    "fastga_he.submodel.propulsion.constraints.pmsm.rpm.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PMSM_TORQUE,
    "fastga_he.submodel.propulsion.constraints.pmsm.torque.enforce",
)
class ConstraintsTorqueEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum torque seen by the motor during the mission is used for
    the sizing, ensuring a fitted design for the torque of each component.
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

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_rating"] = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_max"
        ]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PMSM_RPM,
    "fastga_he.submodel.propulsion.constraints.pmsm.rpm.enforce",
)
class ConstraintsRPMEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum rpm seen by the motor during the mission is used for
    the sizing, ensuring a fitted design for the torque of each component.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_max",
            units="min**-1",
            val=np.nan,
            desc="Maximum value of the motor rpm during the mission",
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


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PMSM_VOLTAGE,
    "fastga_he.submodel.propulsion.constraints.pmsm.voltage.enforce",
)
class ConstraintsVoltageEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum voltage seen by the motor during the mission is used for
    the sizing, ensuring a fitted design for each component.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage_ac_max",
            units="V",
            val=np.nan,
            desc="Maximum value of the peak voltage at the input of the motor",
        )
        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage_caliber",
            val=800.0,
            units="V",
            desc="Max voltage of the motor",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage_caliber",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage_ac_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage_caliber"] = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage_ac_max"
        ]
