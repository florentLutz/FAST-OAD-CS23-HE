# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_mission_power_split import PerformancesMissionPowerSplit
from .perf_electric_node import PerformancesElectricalNode
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

    def setup(self):

        number_of_points = self.options["number_of_points"]
        dc_splitter_id = self.options["dc_splitter_id"]

        self.add_subsystem(
            name="power_split_formatting",
            subsys=PerformancesMissionPowerSplit(
                dc_splitter_id=dc_splitter_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="electrical_node",
            subsys=PerformancesElectricalNode(
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
