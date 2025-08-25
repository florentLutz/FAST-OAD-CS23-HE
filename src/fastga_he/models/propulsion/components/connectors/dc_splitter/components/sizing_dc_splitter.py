# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_dc_splitter_cross_section_area import SizingDCSplitterCrossSectionArea
from .sizing_dc_splitter_cross_section_dimensions import SizingSplitterCrossSectionDimensions
from .sizing_dc_splitter_insulation_thickness import SizingDCSplitterInsulationThickness
from .sizing_dc_splitter_dimensions import SizingDCSplitterDimensions
from .sizing_dc_splitter_weight import SizingDCSplitterWeight
from .sizing_dc_splitter_cg_x import SizingDCSplitterCGX
from .sizing_dc_splitter_cg_y import SizingDCSplitterCGY
from .sizing_dc_splitter_drag import SizingDCSplitterDrag

from .cstr_dc_splitter import ConstraintsDCSplitter

from ..constants import POSSIBLE_POSITION


class SizingDCSplitter(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the DC splitter. Not many
    components, the splitter is mainly a functional component from the point of the performances
    assessment.
    """

    def initialize(self):
        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the DC splitter, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

        # The followong option(s) is/are dummy option(s) to ensure compatibility
        self.options.declare(
            "splitter_mode",
            default="percent_split",
            desc="Mode of the power splitter, should be either percent_split or power_share",
            values=["percent_split", "power_share"],
        )

    def setup(self):
        position = self.options["position"]
        dc_splitter_id = self.options["dc_splitter_id"]

        self.add_subsystem(
            name="constraints_dc_splitter",
            subsys=ConstraintsDCSplitter(dc_splitter_id=dc_splitter_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="splitter_cross_section_area",
            subsys=SizingDCSplitterCrossSectionArea(dc_splitter_id=dc_splitter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="splitter_cross_section_dimensions",
            subsys=SizingSplitterCrossSectionDimensions(dc_splitter_id=dc_splitter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="splitter_insulation",
            subsys=SizingDCSplitterInsulationThickness(dc_splitter_id=dc_splitter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="splitter_dimensions",
            subsys=SizingDCSplitterDimensions(dc_splitter_id=dc_splitter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="splitter_weight",
            subsys=SizingDCSplitterWeight(dc_splitter_id=dc_splitter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="splitter_CG_x",
            subsys=SizingDCSplitterCGX(dc_splitter_id=dc_splitter_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="splitter_CG_y",
            subsys=SizingDCSplitterCGY(dc_splitter_id=dc_splitter_id, position=position),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "dc_splitter_drag_ls" if low_speed_aero else "dc_splitter_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingDCSplitterDrag(
                    dc_splitter_id=dc_splitter_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
