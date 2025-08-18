# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_PMSM_TORQUE,
    SUBMODEL_CONSTRAINTS_PMSM_RPM,
    SUBMODEL_CONSTRAINTS_PMSM_VOLTAGE,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_PMSM_VOLTAGE] = (
    "fastga_he.submodel.propulsion.constraints.ac_pmsm.voltage.ensure"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PMSM_TORQUE,
    "fastga_he.submodel.propulsion.constraints.ac_pmsm.torque.ensure",
)
class ConstraintsTorqueEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum torque seen by the pmsm during the
    mission and the value used for sizing, ensuring each component works below its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_max",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the torque the motor has to provide",
        )
        self.add_input(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
            units="N*m",
            val=np.nan,
            desc="Max continuous torque of the motor",
        )
        self.add_output(
            "constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
            units="N*m",
            val=0.0,
            desc="Respected if <0",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
            wrt=[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_max",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating"] = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_max"]
            - inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
        ] = -1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PMSM_RPM, "fastga_he.submodel.propulsion.constraints.ac_pmsm.rpm.ensure"
)
class ConstraintsRPMEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum rpm seen by the pmsm during the
    mission and the value used for sizing, ensuring each component works below its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_max",
            units="min**-1",
            val=np.nan,
            desc="Maximum value of the motor rpm during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_rating",
            units="min**-1",
            val=np.nan,
            desc="Max continuous rpm of the motor",
        )
        self.add_output(
            "constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_rating",
            units="min**-1",
            val=0.0,
            desc="Respected if <0",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_rating",
            wrt=[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_max",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_rating",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_rating"] = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_max"]
            - inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_rating"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_rating",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_rating",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rpm_rating",
        ] = -1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PMSM_VOLTAGE,
    "fastga_he.submodel.propulsion.constraints.ac_pmsm.voltage.ensure",
)
class ConstraintsVoltageEnsure(om.ExplicitComponent):
    """
    Class that ensure that the maximum voltage seen by the motor during the mission is below the
    one used for the sizing, ensuring each component works below its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_ac_max",
            units="V",
            val=np.nan,
            desc="Maximum value of the peak voltage at the input of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_caliber",
            val=np.nan,
            units="V",
            desc="Max voltage of the motor",
        )

        self.add_output(
            name="constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_caliber",
            val=-0.0,
            units="V",
            desc="Respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_caliber",
            wrt=[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_ac_max",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_caliber",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_caliber"] = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_ac_max"]
            - inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_caliber"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_caliber",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_ac_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_caliber",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":voltage_caliber",
        ] = -1.0
