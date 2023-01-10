# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .sizing_weight import SizingPropellerWeight

from ..constants import SUBMODEL_CONSTRAINTS_PROPELLER


class SizingPropeller(om.Group):
    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        option_propeller_id = {"propeller_id": propeller_id}

        self.add_subsystem(
            name="constraints_propeller",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_PROPELLER, options=option_propeller_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "propeller_wright",  # Like Orville and Wilbur
            SizingPropellerWeight(propeller_id=propeller_id),
            promotes=["data:*"],
        )
