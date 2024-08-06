# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_mission_power_split import PerformancesMissionPowerSplit
from .perf_mission_power_share import PerformancesMissionPowerShare
from .perf_percent_split_equivalent import PerformancesPercentSplitEquivalent
from .perf_electric_node_percent_split import PerformancesElectricalNodePercentSplit
from .perf_maximum import PerformancesMaximum


class PerformancesDCSplitter(om.Group):
    def initialize(self):
        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "splitter_mode",
            default="percent_split",
            desc="Mode of the power splitter, should be either percent_split or power_share",
            values=["percent_split", "power_share"],
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        dc_splitter_id = self.options["dc_splitter_id"]

        if self.options["splitter_mode"] == "percent_split":
            self.add_subsystem(
                name="control_parameter_formatting",
                subsys=PerformancesMissionPowerSplit(
                    dc_splitter_id=dc_splitter_id,
                    number_of_points=number_of_points,
                ),
                promotes=["*"],
            )
        else:
            self.add_subsystem(
                name="control_parameter_formatting",
                subsys=PerformancesMissionPowerShare(
                    dc_splitter_id=dc_splitter_id,
                    number_of_points=number_of_points,
                ),
                promotes=["*"],
            )
            self.add_subsystem(
                name="percent_split_equivalent",
                subsys=PerformancesPercentSplitEquivalent(
                    number_of_points=number_of_points,
                ),
                promotes=["*"],
            )
        self.add_subsystem(
            name="electrical_node",
            subsys=PerformancesElectricalNodePercentSplit(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="maximum",
            subsys=PerformancesMaximum(
                dc_splitter_id=dc_splitter_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
