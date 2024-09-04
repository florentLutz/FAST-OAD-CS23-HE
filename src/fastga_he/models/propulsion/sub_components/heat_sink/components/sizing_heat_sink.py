# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_heat_sink_tube_length import SizingHeatSinkTubeLength
from .sizing_heat_sink_tube_mass_flow import SizingHeatSinkTubeMassFlow
from .sizing_heat_sink_coolant_prandtl import SizingHeatSinkCoolantPrandtl
from .sizing_heat_sink_tube_inner_diameter import (
    SizingHeatSinkTubeInnerDiameter,
)
from .sizing_heat_sink_tube_outer_diameter import (
    SizingHeatSinkTubeOuterDiameter,
)
from .sizing_heat_sink_tube_weight import SizingHeatSinkTubeWeight
from .sizing_heat_sink_height import SizingHeatSinkHeight
from .sizing_heat_sink_weight import SizingHeatSinkWeight


class SizingHeatSink(om.Group):
    """
    Class to regroup all the computation related to the sizing of the heat sink, to make it
    easier to deactivate for when components will be taken on shelf.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use a heatsink",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_subsystem(
            "heat_sink_tube_length",
            SizingHeatSinkTubeLength(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_max_mass_flow",
            SizingHeatSinkTubeMassFlow(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            "coolant_prandtl",
            SizingHeatSinkCoolantPrandtl(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_tube_inner_diameter",
            SizingHeatSinkTubeInnerDiameter(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_tube_outer_diameter",
            SizingHeatSinkTubeOuterDiameter(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_tube_weight",
            SizingHeatSinkTubeWeight(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_height",
            SizingHeatSinkHeight(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_weight",
            SizingHeatSinkWeight(prefix=prefix),
            promotes=["*"],
        )
