# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_ICE_SL_POWER


class ConstraintsICE(om.Group):
    """
    Class that gather the different constraints for the ICE be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):

        ice_id = self.options["ice_id"]

        option_ice_id = {"ice_id": ice_id}

        self.add_subsystem(
            name="constraints_SL_power",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_ICE_SL_POWER, options=option_ice_id
            ),
            promotes=["*"],
        )
