# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.sizing_turboshaft_uninstalled_weight import SizingTurboshaftUninstalledWeight
from ..components.sizing_turboshaft_weight import SizingTurboshaftWeight
from ..components.sizing_turboshaft_dimensions import SizingTurboshaftDimensions
from ..components.sizing_turboshaft_nacelle_dimensions import SizingTurboshaftNacelleDimensions
from ..components.sizing_turboshaft_nacelle_wet_area import SizingTurboshaftNacelleWetArea
from ..components.sizing_turboshaft_drag import SizingTurboshaftDrag
from ..components.sizing_turboshaft_cg_x import SizingTurboshaftCGX
from ..components.sizing_turboshaft_cg_y import SizingTurboshaftCGY

from ..components.cstr_turboshaft import ConstraintsTurboshaft

from ..constants import POSSIBLE_POSITION


class SizingTurboshaft(om.Group):
    def initialize(self):

        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the turboshaft, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        turboshaft_id = self.options["turboshaft_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_turboshaft",
            subsys=ConstraintsTurboshaft(turboshaft_id=turboshaft_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="uninstalled_weight",
            subsys=SizingTurboshaftUninstalledWeight(turboshaft_id=turboshaft_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="installed_weight",
            subsys=SizingTurboshaftWeight(turboshaft_id=turboshaft_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="turboshaft_dimensions",
            subsys=SizingTurboshaftDimensions(turboshaft_id=turboshaft_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="nacelle_dimensions",
            subsys=SizingTurboshaftNacelleDimensions(
                turboshaft_id=turboshaft_id, position=position
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="nacelle_wet_area",
            subsys=SizingTurboshaftNacelleWetArea(turboshaft_id=turboshaft_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="turboshaft_cg_x",
            subsys=SizingTurboshaftCGX(turboshaft_id=turboshaft_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="turboshaft_cg_y",
            subsys=SizingTurboshaftCGY(turboshaft_id=turboshaft_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "turboshaft_drag_ls" if low_speed_aero else "turboshaft_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingTurboshaftDrag(
                    turboshaft_id=turboshaft_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
