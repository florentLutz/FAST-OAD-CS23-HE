# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_bus_cross_section_area import SizingBusBarCrossSectionArea
from .sizing_bus_bar_cross_section_dimensions import SizingBusBarCrossSectionDimensions
from .sizing_insulation_thickness import SizingBusBarInsulationThickness
from .sizing_bus_dimensions import SizingBusBarDimensions
from .sizing_bus_bar_weight import SizingBusBarWeight
from .sizing_conductor_self_inductance import SizingBusBarSelfInductance
from .sizing_conductor_mutual_inductance import SizingBusBarMutualInductance
from .sizing_dc_bus_cg_x import SizingDCBusCGX
from .sizing_dc_bus_cg_y import SizingDCBusCGY
from .sizing_dc_bus_drag import SizingDCBusDrag

from .cstr_dc_bus import ConstraintsDCBus

from ..constants import POSSIBLE_POSITION


class SizingDCBus(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the DC Bus.
    """

    def initialize(self):
        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the DC bus, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

        # The followong option(s) is/are dummy option(s) to prevent error
        self.options.declare(
            name="number_of_inputs",
            default=1,
            types=int,
            desc="Number of connections at the input of the bus",
            allow_none=False,
        )
        self.options.declare(
            name="number_of_outputs",
            default=1,
            types=int,
            desc="Number of connections at the output of the bus",
            allow_none=False,
        )

    def setup(self):
        dc_bus_id = self.options["dc_bus_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_dc_bus",
            subsys=ConstraintsDCBus(dc_bus_id=dc_bus_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="conductor_cross_section_area",
            subsys=SizingBusBarCrossSectionArea(dc_bus_id=dc_bus_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="conductor_cross_section_dimensions",
            subsys=SizingBusBarCrossSectionDimensions(dc_bus_id=dc_bus_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="insulation_thickness",
            subsys=SizingBusBarInsulationThickness(dc_bus_id=dc_bus_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="bus_bar_dimensions",
            subsys=SizingBusBarDimensions(dc_bus_id=dc_bus_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="bus_bar_weight",
            subsys=SizingBusBarWeight(dc_bus_id=dc_bus_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="bus_bar_self_inductance",
            subsys=SizingBusBarSelfInductance(dc_bus_id=dc_bus_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="bus_bar_mutual_inductance",
            subsys=SizingBusBarMutualInductance(dc_bus_id=dc_bus_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="bus_CG_x",
            subsys=SizingDCBusCGX(dc_bus_id=dc_bus_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="bus_CG_y",
            subsys=SizingDCBusCGY(dc_bus_id=dc_bus_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "dc_bus_drag_ls" if low_speed_aero else "dc_bus_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingDCBusDrag(
                    dc_bus_id=dc_bus_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
