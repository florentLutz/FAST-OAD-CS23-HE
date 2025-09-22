# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_SM_PMSM_TORQUE,
    SUBMODEL_CONSTRAINTS_SM_PMSM_RPM,
    SUBMODEL_CONSTRAINTS_SM_PMSM_VOLTAGE,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_SM_PMSM_VOLTAGE] = (
    "fastga_he.submodel.propulsion.constraints.sm_pmsm.voltage.ensure"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_SM_PMSM_TORQUE,
    "fastga_he.submodel.propulsion.constraints.sm_pmsm.torque.ensure",
)
class ConstraintsTorqueEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum torque seen by the motor during the
    mission and the value used for sizing, ensuring each component works below its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the torque the motor has to provide",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating",
            units="N*m",
            val=np.nan,
            desc="Max continuous torque of the motor",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating",
            units="N*m",
            val=0.0,
            desc="Respected if <0",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max"]
            - inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating"]
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_SM_PMSM_RPM,
    "fastga_he.submodel.propulsion.constraints.sm_pmsm.rpm.ensure",
)
class ConstraintsRPMEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum rpm seen by the motor during the
    mission and the value used for sizing, ensuring each component works below its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max",
            units="min**-1",
            val=np.nan,
            desc="Maximum value of the motor rpm during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            units="min**-1",
            val=np.nan,
            desc="Max continuous rpm of the motor",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            units="min**-1",
            val=0.0,
            desc="Respected if <0",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max"]
            - inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_SM_PMSM_VOLTAGE,
    "fastga_he.submodel.propulsion.constraints.sm_pmsm.voltage.ensure",
)
class ConstraintsVoltageEnsure(om.ExplicitComponent):
    """
    Class that ensure that the maximum voltage seen by the motor during the mission is below the
    one used for the sizing, ensuring each component works below its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_ac_max",
            units="V",
            val=np.nan,
            desc="Maximum value of the peak voltage at the input of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_caliber",
            val=np.nan,
            units="V",
            desc="Max voltage of the motor",
        )

        self.add_output(
            name="constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_caliber",
            val=-0.0,
            units="V",
            desc="Respected if <0",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_caliber",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_ac_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_caliber",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_caliber",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs[
            "constraints:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_ac_max"]
            - inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_caliber"]
        )
