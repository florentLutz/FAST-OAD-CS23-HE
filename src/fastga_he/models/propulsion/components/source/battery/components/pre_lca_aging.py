#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from .pre_lca_depth_of_discharge import PreLCABatteryDepthOfDischarge
from .pre_lca_cyclic_aging_dod_effect import PreLCABatteryCyclicAgingDODEffect
from .pre_lca_life_cycle_cyclic import PreLCABatteryCyclicAging

from ..constants import SERVICE_BATTERY_LIFESPAN

# I want the default behaviour to be that the lifespan is an input
oad.RegisterSubmodel.active_models[SERVICE_BATTERY_LIFESPAN] = None


@oad.RegisterSubmodel(
    SERVICE_BATTERY_LIFESPAN, "fastga_he.submodel.propulsion.battery.lifespan.legacy_aging_model"
)
class PreLCABatteryAging(om.Group):
    """
    Group that contain all the model pertaining to the battery aging model
    """

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.add_subsystem(
            name="depth_of_discharge",
            subsys=PreLCABatteryDepthOfDischarge(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="dod_effect",
            subsys=PreLCABatteryCyclicAgingDODEffect(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="cyclic_aging",
            subsys=PreLCABatteryCyclicAging(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
