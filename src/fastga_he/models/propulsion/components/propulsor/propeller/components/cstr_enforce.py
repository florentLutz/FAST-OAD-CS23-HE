# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_PROPELLER_TORQUE, SUBMODEL_CONSTRAINTS_PROPELLER_RPM

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_PROPELLER_TORQUE] = (
    "fastga_he.submodel.propulsion.constraints.propeller.torque.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_PROPELLER_RPM] = (
    "fastga_he.submodel.propulsion.constraints.propeller.rpm.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PROPELLER_TORQUE,
    "fastga_he.submodel.propulsion.constraints.propeller.torque.enforce",
)
class ConstraintsTorqueEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum torque seen by the propeller during the mission is used
    for the sizing, ensuring a fitted design of each component.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]

        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the propeller torque during performances assessment",
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            units="N*m",
            val=250.0,
            desc="Maximum value of the propeller torque used for the sizing",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            wrt="data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating"] = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max"]
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PROPELLER_RPM,
    "fastga_he.submodel.propulsion.constraints.propeller.rpm.enforce",
)
class ConstraintsRPMEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum RPM seen by the propeller during the mission is used
    for the sizing, ensuring a fitted design of each component.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]

        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max",
            units="min**-1",
            val=np.nan,
            desc="Maximum value of the propeller RPM during performances assessment",
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            units="min**-1",
            val=1000.0,
            desc="The propeller RPM rating used for the sizing",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            wrt="data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating"] = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max"]
        )
