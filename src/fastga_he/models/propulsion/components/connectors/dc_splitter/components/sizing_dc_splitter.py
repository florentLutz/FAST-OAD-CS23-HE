# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .sizing_dc_splitter_weight import SizingDCSplitterWeight
from .sizing_dc_splitter_cg import SizingDCSplitterCG
from .sizing_dc_splitter_drag import SizingDCSplitterDrag

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

    def setup(self):

        position = self.options["position"]
        dc_splitter_id = self.options["dc_splitter_id"]

        self.add_subsystem(
            name="splitter_weight",
            subsys=SizingDCSplitterWeight(dc_splitter_id=dc_splitter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="splitter_CG",
            subsys=SizingDCSplitterCG(dc_splitter_id=dc_splitter_id, position=position),
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
