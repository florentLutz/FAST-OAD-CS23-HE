# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_DC_BUS

from ..components.sizing_bus_cross_section_area import SizingBusBarCrossSectionArea
from ..components.sizing_bus_bar_cross_section_dimensions import SizingBusBarCrossSectionDimensions
from ..components.sizing_insulation_thickness import SizingBusBarInsulationThickness
from ..components.sizing_bus_dimensions import SizingBusBarDimensions
from ..components.sizing_bus_bar_weight import SizingBusBarWeight
from ..components.sizing_conductor_self_inductance import SizingBusBarSelfInductance
from ..components.sizing_conductor_mutual_inductance import SizingBusBarMutualInductance


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

    def setup(self):
        dc_bus_id = self.options["dc_bus_id"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        option_bus_id = {"dc_bus_id": dc_bus_id}

        self.add_subsystem(
            name="constraints_dc_bus",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_BUS, options=option_bus_id
            ),
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
