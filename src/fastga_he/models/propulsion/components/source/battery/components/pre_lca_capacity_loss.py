#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_depth_of_discharge import PreLCABatteryDepthOfDischarge
from .pre_lca_cyclic_aging_dod_effect import PreLCABatteryCyclicAgingDODEffect
from .pre_lca_life_cycle_cyclic import PreLCABatteryCyclicAging
from .pre_lca_total_aging import PreLCABatteryTotalAging
from .pre_lca_time_between_cycles import PreLCABatteryTimeBetweenCycles
from .pre_lca_calendar_aging_soc_effect import PreLCABatteryCalendarAgingSOCEffect
from .pre_lca_life_cycle_calendar import PreLCABatteryCalendarAging


class PreLCABatteryCapacityLoss(om.Group):
    """
    Group that contain all the model pertaining to the battery capacity loss model
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
        self.add_subsystem(
            name="time_between_cycles",
            subsys=PreLCABatteryTimeBetweenCycles(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="soc_effect",
            subsys=PreLCABatteryCalendarAgingSOCEffect(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="calendar_aging",
            subsys=PreLCABatteryCalendarAging(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="total_aging",
            subsys=PreLCABatteryTotalAging(),
            promotes=["*"],
        )
