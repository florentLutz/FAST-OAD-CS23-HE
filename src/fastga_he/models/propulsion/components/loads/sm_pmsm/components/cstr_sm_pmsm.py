# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import (
    SUBMODEL_CONSTRAINTS_SM_PMSM_TORQUE,
    SUBMODEL_CONSTRAINTS_SM_PMSM_RPM,
    SUBMODEL_CONSTRAINTS_SM_PMSM_VOLTAGE,
)


class ConstraintsSMPMSM(om.Group):
    """
    Class that gather the different ensure or enforce constraints for the surface mounted PMSM.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        option_motor_id = {"motor_id": motor_id}

        self.add_subsystem(
            name="constraints_torque_pmsm",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_SM_PMSM_TORQUE, options=option_motor_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_rpm_pmsm",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_SM_PMSM_RPM, options=option_motor_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_voltage_pmsm",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_SM_PMSM_VOLTAGE, options=option_motor_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_for_power_rate",
            subsys=ConstraintPMSMPowerRateMission(motor_id=motor_id),
            promotes=["*"],
        )


class ConstraintPMSMPowerRateMission(om.ExplicitComponent):
    """
    This class will define the value of the maximum power we use to get the power rate inside the
    mission, it is mandatory that we compute it outside the mission when sizing the power train
    or else when recomputing the wing area it will be stuck at one which we don't want. It's
    nothing complex but has to be done outside the mission with consistent naming.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_power_max",
            units="kW",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_power_rating",
            units="MW",
            val=1.4326,
            desc="Value of the maximum power the PMSM can provide, used for sizing",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", val=0.001)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_power_rating"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_power_max"]
            / 1000.0
        )
