# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .cstr_dc_sspc import ConstraintsDCSSPC

from .sizing_resistance_scaling import SizingDCSSPCResistanceScaling
from .sizing_reference_resistance import SizingDCSSPCResistances
from .sizing_efficiency import SizingDCSSPCEfficiency
from .sizing_weight import SizingDCSSPCWeight
from .sizing_dc_sspc_cg_x import SizingDCSSPCCGX
from .sizing_dc_sspc_cg_y import SizingDCSSPCCGY
from .sizing_dc_sspc_drag import SizingDCSSPCDrag

from ..constants import POSSIBLE_POSITION


class SizingDCSSPC(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the DC SSPC.
    """

    def initialize(self):

        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the DC SSPC, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        dc_sspc_id = self.options["dc_sspc_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_dc_sspc",
            subsys=ConstraintsDCSSPC(dc_sspc_id=dc_sspc_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="resistance_scaling",
            subsys=SizingDCSSPCResistanceScaling(dc_sspc_id=dc_sspc_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="resistance",
            subsys=SizingDCSSPCResistances(dc_sspc_id=dc_sspc_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="efficiency",
            subsys=SizingDCSSPCEfficiency(dc_sspc_id=dc_sspc_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="weight",
            subsys=SizingDCSSPCWeight(dc_sspc_id=dc_sspc_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="sspc_CG_x",
            subsys=SizingDCSSPCCGX(dc_sspc_id=dc_sspc_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="sspc_CG_y",
            subsys=SizingDCSSPCCGY(dc_sspc_id=dc_sspc_id, position=position),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "dc_sspc_drag_ls" if low_speed_aero else "dc_sspc_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingDCSSPCDrag(
                    dc_sspc_id=dc_sspc_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
