# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_PLANETARY_GEAR_TORQUE


class ConstraintsPlanetaryGear(om.Group):
    """
    Class that gather the different constraints for the planetary gear be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the planetary gear",
            allow_none=False,
        )

    def setup(self):
        planetary_gear_id = self.options["planetary_gear_id"]

        option_planetary_gear_id = {"planetary_gear_id": planetary_gear_id}

        self.add_subsystem(
            name="constraints_torque_planetary_gear",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_PLANETARY_GEAR_TORQUE, options=option_planetary_gear_id
            ),
            promotes=["*"],
        )
