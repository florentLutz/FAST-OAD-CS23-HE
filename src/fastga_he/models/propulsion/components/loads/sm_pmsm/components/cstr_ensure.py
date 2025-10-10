# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_SM_PMSM_TORQUE,
    SUBMODEL_CONSTRAINTS_SM_PMSM_RPM,
    SUBMODEL_CONSTRAINTS_SM_PMSM_CURRENT_DENSITY,
    SUBMODEL_CONSTRAINTS_SM_PMSM_TANGENTIAL_STRESS,
    SUBMODEL_CONSTRAINTS_SM_PMSM_ELECTROMAGNETIC_TORQUE,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad


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
    SUBMODEL_CONSTRAINTS_SM_PMSM_ELECTROMAGNETIC_TORQUE,
    "fastga_he.submodel.propulsion.constraints.sm_pmsm.electromagnetic_torque.ensure",
)
class ConstraintsElectromagneticTorqueEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum electromagnetic torque seen by the motor
    during the mission and the value used for sizing, ensuring each component works below its
    maximum.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_max",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the electromagnetic torque the motor has to provide",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_rating",
            units="N*m",
            val=np.nan,
            desc="Max electromagnetic torque of the motor",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":electromagnetic_torque_rating",
            units="N*m",
            val=0.0,
            desc="Respected if <0",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":electromagnetic_torque_rating",
            wrt="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":electromagnetic_torque_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":electromagnetic_torque_rating",
            wrt="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":electromagnetic_torque_rating",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs[
            "constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":electromagnetic_torque_rating"
        ] = (
            inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:SM_PMSM:"
                + motor_id
                + ":electromagnetic_torque_rating"
            ]
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_SM_PMSM_CURRENT_DENSITY,
    "fastga_he.submodel.propulsion.constraints.sm_pmsm.current_density.ensure",
)
class ConstraintsCurrentDensityEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum phase current density seen by the
    motor during the mission and the value used for sizing, ensuring each component works below
    its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_max",
            units="kA/m**2",
            val=np.nan,
            desc="Maximum value of the motor phase current density during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_caliber",
            units="kA/m**2",
            val=np.nan,
            desc="Max phase current density of the motor",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":current_density_ac_caliber",
            units="kA/m**2",
            val=0.0,
            desc="Respected if <0",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":current_density_ac_caliber",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":current_density_ac_caliber",
            wrt="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":current_density_ac_caliber",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs[
            "constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":current_density_ac_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_max"]
            - inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_caliber"
            ]
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_SM_PMSM_TANGENTIAL_STRESS,
    "fastga_he.submodel.propulsion.constraints.sm_pmsm.tangential_stress.ensure",
)
class ConstraintsTangentialStressEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum tangential stress applied on the rotor
    during the mission and the value used for sizing, ensuring each component works below
    its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_max",
            units="MPa",
            val=np.nan,
            desc="Maximum value of the rotor tangential stress during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber",
            units="MPa",
            val=np.nan,
            desc="Max tangential stress of the rotor",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":tangential_stress_caliber",
            units="MPa",
            val=0.0,
            desc="Respected if <0",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":tangential_stress_caliber",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":tangential_stress_caliber",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs[
            "constraints:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":tangential_stress_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_max"]
            - inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber"
            ]
        )
