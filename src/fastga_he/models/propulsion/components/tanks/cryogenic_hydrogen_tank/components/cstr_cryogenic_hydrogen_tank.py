# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_CRYOGENIC_HYDROGEN_TANK_CAPACITY


class ConstraintsCryogenicHydrogenTank(om.Group):
    """
    Class that gather the different constraints for the fuel tank, be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

    def setup(self):
        option_cryogenic_hydrogen_tank_id = {
            "cryogenic_hydrogen_tank_id": self.options["cryogenic_hydrogen_tank_id"]
        }

        self.add_subsystem(
            name="constraints_hydrogen_gas_tank",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_CRYOGENIC_HYDROGEN_TANK_CAPACITY,
                options=option_cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )
