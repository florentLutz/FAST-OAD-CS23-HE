# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import (
    SUBMODEL_CONSTRAINTS_AUX_LOAD_POWER,
)


class ConstraintsAuxLoad(om.Group):
    """
    Class that gather the different constraints for the aux load be they ensure or enforce.
    """

    def initialize(self):

        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )

    def setup(self):

        aux_load_id = self.options["aux_load_id"]

        option_aux_load_id = {"aux_load_id": aux_load_id}

        self.add_subsystem(
            name="constraints_power_aux_load",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_AUX_LOAD_POWER, options=option_aux_load_id
            ),
            promotes=["*"],
        )
