# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_heat_sink_dimension import SizingInverterHeatSinkDimension
from .sizing_heat_sink_tube_length import SizingInverterHeatSinkTubeLength
from .sizing_heat_sink_tube_mass_flow import SizingInverterHeatSinkTubeMassFlow
from .sizing_heat_sink_coolant_prandtl import SizingInverterHeatSinkCoolantPrandtl
from .sizing_heat_sink_tube_inner_diameter import (
    SizingInverterHeatSinkTubeInnerDiameter,
)
from .sizing_heat_sink_tube_outer_diameter import (
    SizingInverterHeatSinkTubeOuterDiameter,
)
from .sizing_heat_sink_tube_weight import SizingInverterHeatSinkTubeWeight
from .sizing_height_heat_sink import SizingInverterHeatSinkHeight
from .sizing_heat_sink_weight import SizingInverterHeatSinkWeight


class SizingHeatSink(om.Group):
    """
    Class to regroup all the computation related to the sizing of the heat sink, to make it
    easier to deactivate for when components will be taken on shelf.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]

        self.add_subsystem(
            "heat_sink_dimensions",
            SizingInverterHeatSinkDimension(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_tube_length",
            SizingInverterHeatSinkTubeLength(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_max_mass_flow",
            SizingInverterHeatSinkTubeMassFlow(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "coolant_prandtl",
            SizingInverterHeatSinkCoolantPrandtl(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_tube_inner_diameter",
            SizingInverterHeatSinkTubeInnerDiameter(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_tube_outer_diameter",
            SizingInverterHeatSinkTubeOuterDiameter(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_tube_weight",
            SizingInverterHeatSinkTubeWeight(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_height",
            SizingInverterHeatSinkHeight(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_weight",
            SizingInverterHeatSinkWeight(inverter_id=inverter_id),
            promotes=["*"],
        )
