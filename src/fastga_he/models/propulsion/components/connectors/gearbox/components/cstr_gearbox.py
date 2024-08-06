# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_GEARBOX_TORQUE


class ConstraintsGearbox(om.Group):
    """
    Class that gather the different constraints for the gearbox be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )

    def setup(self):
        gearbox_id = self.options["gearbox_id"]

        option_gearbox_id = {"gearbox_id": gearbox_id}

        self.add_subsystem(
            name="constraints_torque_gearbox",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_GEARBOX_TORQUE, options=option_gearbox_id
            ),
            promotes=["*"],
        )
