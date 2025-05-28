# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_PROPELLER_TORQUE, SUBMODEL_CONSTRAINTS_PROPELLER_RPM

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PROPELLER_TORQUE,
    "fastga_he.submodel.propulsion.constraints.propeller.torque.ensure",
)
class ConstraintsTorqueEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum torque seen by the propeller during the
    mission and the value used for sizing, ensuring each component works below its maxima.
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
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the propeller torque used for the sizing",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            units="N*m",
            val=-1.0,
            desc="Respected if negative",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            wrt=[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        outputs[
            "constraints:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating"
        ] = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max"]
            - inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        partials[
            "constraints:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
        ] = -1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PROPELLER_RPM,
    "fastga_he.submodel.propulsion.constraints.propeller.rpm.ensure",
)
class ConstraintsRPMEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum RPM seen by the propeller during the
    mission and the value used for sizing, ensuring each component works below its maxima.
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
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            units="min**-1",
            val=np.nan,
            desc="Maximum value of the propeller RPM used for the sizing",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            units="min**-1",
            val=-0.0,
            desc="Respected if negative",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            wrt=[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        outputs[
            "constraints:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating"
        ] = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max"]
            - inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        partials[
            "constraints:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
        ] = -1.0
