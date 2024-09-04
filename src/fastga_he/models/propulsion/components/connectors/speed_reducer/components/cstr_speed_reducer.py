# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_SPEED_REDUCER_TORQUE


class ConstraintsSpeedReducer(om.Group):
    """
    Class that gather the different constraints for the speed reducer be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="speed_reducer_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )

    def setup(self):
        speed_reducer_id = self.options["speed_reducer_id"]

        option_speed_reducer_id = {"speed_reducer_id": speed_reducer_id}

        self.add_subsystem(
            name="constraints_torque_speed_reducer",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_SPEED_REDUCER_TORQUE, options=option_speed_reducer_id
            ),
            promotes=["*"],
        )
