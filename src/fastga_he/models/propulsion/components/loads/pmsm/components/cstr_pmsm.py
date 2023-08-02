# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import (
    SUBMODEL_CONSTRAINTS_PMSM_TORQUE,
    SUBMODEL_CONSTRAINTS_PMSM_RPM,
    SUBMODEL_CONSTRAINTS_PMSM_VOLTAGE,
)


class ConstraintsPMSM(om.Group):
    """
    Class that gather the different constraints for the PMSM be they ensure or enforce.
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
                SUBMODEL_CONSTRAINTS_PMSM_TORQUE, options=option_motor_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_rpm_pmsm",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_PMSM_RPM, options=option_motor_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_voltage_pmsm",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_PMSM_VOLTAGE, options=option_motor_id
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
    nothing complex but we need it done outside of the mission and we need consistent naming.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_max",
            units="kW",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_rating",
            units="kW",
            val=42000.0,
            desc="Value of the maximum power the PMSM can provide, used for sizing",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_rating"] = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_max"
        ]
