# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad
from fastoad_cs25.models.weight.mass_breakdown.constants import SERVICE_PROPULSION_MASS
from .b1_power_train_mass import PowerTrainMass

POWERTRAIN_MASS = "fastga_he.submodel.weight.mass.propulsion.power_train.rta"


@oad.RegisterSubmodel(SERVICE_PROPULSION_MASS, POWERTRAIN_MASS)
class PowerTrainMassRTA(om.Group):
    """
    This submodel serves to prevent variable conflicts of the masses related to the powertrain.
    """

    def initialize(self):
        # Needed even if it isn't used because the original one has that option ...
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem(
            "propulsion_mass",
            PowerTrainMass(propulsion_id=self.options["propulsion_id"]),
            promotes=["data:*"],
        )
