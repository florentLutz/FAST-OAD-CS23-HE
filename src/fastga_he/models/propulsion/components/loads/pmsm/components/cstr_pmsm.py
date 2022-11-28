# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_PMSM_TORQUE, SUBMODEL_CONSTRAINTS_PMSM_RPM


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
