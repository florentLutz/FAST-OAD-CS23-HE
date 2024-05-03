# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_rpm_in import PerformancesRPMIn
from .perf_mission_power_split import PerformancesMissionPowerSplit
from .perf_mission_power_share import PerformancesMissionPowerShare
from .perf_percent_split_equivalent import PerformancesPercentSplitEquivalent
from .perf_shaft_power_in import PerformancesShaftPowerIn
from .perf_torque_out import PerformancesTorqueOut
from .perf_torque_in import PerformancesTorqueIn
from .perf_maximum import PerformancesMaximum


class PerformancesPlanetaryGear(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the planetary gear",
            allow_none=False,
        )
        self.options.declare(
            "gear_mode",
            default="percent_split",
            desc="Mode of the planetary gear, should be either percent_split or power_share",
            values=["percent_split", "power_share"],
        )

    def setup(self):

        planetary_gear_id = self.options["planetary_gear_id"]
        number_of_points = self.options["number_of_points"]

        if self.options["gear_mode"] == "percent_split":
            self.add_subsystem(
                name="control_parameter_formatting",
                subsys=PerformancesMissionPowerSplit(
                    planetary_gear_id=planetary_gear_id,
                    number_of_points=number_of_points,
                ),
                promotes=["*"],
            )
        else:
            self.add_subsystem(
                name="control_parameter_formatting",
                subsys=PerformancesMissionPowerShare(
                    planetary_gear_id=planetary_gear_id,
                    number_of_points=number_of_points,
                ),
                promotes=["*"],
            )
            self.add_subsystem(
                name="percent_split_equivalent",
                subsys=PerformancesPercentSplitEquivalent(
                    planetary_gear_id=planetary_gear_id,
                    number_of_points=number_of_points,
                ),
                promotes=["*"],
            )
        self.add_subsystem(
            "rpm_in",
            PerformancesRPMIn(
                number_of_points=number_of_points, planetary_gear_id=planetary_gear_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "shaft_power_in",
            PerformancesShaftPowerIn(
                number_of_points=number_of_points, planetary_gear_id=planetary_gear_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "torque_in",
            PerformancesTorqueIn(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "torque_out",
            PerformancesTorqueOut(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(
                number_of_points=number_of_points, planetary_gear_id=planetary_gear_id
            ),
            promotes=["*"],
        )
