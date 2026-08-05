# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .components_ducted_fan.constants import POSSIBLE_POSITION
from .components_ducted_fan.sizing_ducted_fan_weight import SizingDuctedFanWeight
from .components_ducted_fan.sizing_ducted_fan_cg_x import SizingDuctedFanCGX
from .components_ducted_fan.sizing_ducted_fan_cg_y import SizingDuctedFanCGY
from .components_ducted_fan.sizing_ducted_fan_drag import SizingDuctedFanDrag
from .components_ducted_fan.sizing_ducted_fan_ref_chord import SizingDuctedFanReferenceChord

# TODO: not yet converted -- ConstraintsDuctedFan needs oad.RegisterSubmodel +
# SUBMODEL_CONSTRAINTS_* constants (see ..components.cstr_propeller for the propeller's pattern)
# that don't exist for the ducted fan yet. Commented out so this group runs today with what has
# been converted; add it back in once built.
# from .components_ducted_fan.cstr_ducted_fan import ConstraintsDuctedFan


class SizingDuctedFan(om.Group):
    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the ducted fan, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]
        position = self.options["position"]

        # TODO: add constraints_ducted_fan (ConstraintsDuctedFan) here once built, mirroring
        # ..components.sizing_propeller's constraints_propeller subsystem.

        self.add_subsystem(
            "ducted_fan_weight",
            SizingDuctedFanWeight(ducted_fan_id=ducted_fan_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "ducted_fan_CG_x",
            SizingDuctedFanCGX(ducted_fan_id=ducted_fan_id, position=position),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "ducted_fan_CG_y",
            SizingDuctedFanCGY(ducted_fan_id=ducted_fan_id, position=position),
            promotes=["data:*"],
        )
        # Needed by the slipstream delta_Cl model in slipstream_ducted_fan_new.py (turns the fan
        # diameter into a blown wing-area fraction).
        self.add_subsystem(
            "ducted_fan_ref_chord",
            SizingDuctedFanReferenceChord(ducted_fan_id=ducted_fan_id),
            promotes=["data:*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "ducted_fan_drag_ls" if low_speed_aero else "ducted_fan_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingDuctedFanDrag(
                    ducted_fan_id=ducted_fan_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
