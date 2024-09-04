# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_PROPELLER_TORQUE


class ConstraintsPropeller(om.Group):
    """
    Class that gather the different constraints for the propeller be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        option_propeller_id = {"propeller_id": self.options["propeller_id"]}

        self.add_subsystem(
            name="constraints_torque_propeller",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_PROPELLER_TORQUE, options=option_propeller_id
            ),
            promotes=["*"],
        )
