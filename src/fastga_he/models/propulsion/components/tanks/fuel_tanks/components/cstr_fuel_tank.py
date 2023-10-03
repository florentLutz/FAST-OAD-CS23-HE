# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_FUEL_TANK_CAPACITY


class ConstraintsFuelTank(om.Group):
    """
    Class that gather the different constraints for the fuel tank, be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):

        option_fuel_tank_id = {"fuel_tank_id": self.options["fuel_tank_id"]}

        self.add_subsystem(
            name="constraints_soc_battery",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_FUEL_TANK_CAPACITY, options=option_fuel_tank_id
            ),
            promotes=["*"],
        )
