# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_HYDROGEN_GAS_TANK_CAPACITY


class ConstraintsHydrogenGasTank(om.Group):
    """
    Class that gather the different constraints for the fuel tank, be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="hydrogen_gas_tank_id",
            default=None,
            desc="Identifier of the hydrogen gas tank",
            allow_none=False,
        )

    def setup(self):
        option_hydrogen_gas_tank_id = {"hydrogen_gas_tank_id": self.options["hydrogen_gas_tank_id"]}

        self.add_subsystem(
            name="constraints_hydrogen_gas_tank",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_HYDROGEN_GAS_TANK_CAPACITY, options=option_hydrogen_gas_tank_id
            ),
            promotes=["*"],
        )
