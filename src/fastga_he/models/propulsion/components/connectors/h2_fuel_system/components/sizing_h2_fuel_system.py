# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from ..components.sizing_h2_fuel_system_length import SizingH2FuelSystemLength
from ..components.sizing_h2_fuel_system_cg_x import SizingH2FuelSystemCGX
from ..components.sizing_h2_fuel_system_cg_y import SizingH2FuelSystemCGY
from ..components.sizing_h2_fuel_system_inner_diameter import SizingH2FuelSystemInnerDiameter
from ..components.sizing_h2_fuel_system_cross_section import SizingH2FuelSystemCrossSectionDimension
from ..components.sizing_h2_fuel_system_relative_roughness import (
    SizingH2FuelSystemRelativeRoughness,
)
from ..components.sizing_h2_fuel_system_weight import SizingH2FuelSystemWeight
from ..components.sizing_h2_fuel_system_drag import SizingH2FuelSystemDrag

from ..constants import POSSIBLE_POSITION


class SizingH2FuelSystem(om.Group):
    """
    Class that regroups all subcomponents for the sizing of the hydrogen fuel system.
    """

    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="from_rear_to_center",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the hydrogen fuel system, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="h2_fuel_system_length",
            subsys=SizingH2FuelSystemLength(h2_fuel_system_id=h2_fuel_system_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="h2_fuel_system_inner_diameter",
            subsys=SizingH2FuelSystemInnerDiameter(h2_fuel_system_id=h2_fuel_system_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="h2_fuel_system_cross_section",
            subsys=SizingH2FuelSystemCrossSectionDimension(h2_fuel_system_id=h2_fuel_system_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="h2_fuel_system_cg_x",
            subsys=SizingH2FuelSystemCGX(h2_fuel_system_id=h2_fuel_system_id, position=position),
            promotes=["*"],
        )

        self.add_subsystem(
            name="h2_fuel_system_cg_y",
            subsys=SizingH2FuelSystemCGY(h2_fuel_system_id=h2_fuel_system_id, position=position),
            promotes=["*"],
        )

        self.add_subsystem(
            name="h2_fuel_system_mass",
            subsys=SizingH2FuelSystemWeight(h2_fuel_system_id=h2_fuel_system_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="h2_fuel_system_relative_roughness",
            subsys=SizingH2FuelSystemRelativeRoughness(h2_fuel_system_id=h2_fuel_system_id),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "fuel_system_drag_ls" if low_speed_aero else "fuel_system_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingH2FuelSystemDrag(
                    h2_fuel_system_id=h2_fuel_system_id,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
