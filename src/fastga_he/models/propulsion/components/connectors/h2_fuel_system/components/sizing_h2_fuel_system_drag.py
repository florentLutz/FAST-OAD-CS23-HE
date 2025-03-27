# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class SizingH2FuelSystemDrag(om.ExplicitComponent):
    """
    Computation of the drag coefficient of the hydrogen fuel system based on its position. Remains
    0.0 as the system is assumed inside the fuselage or inside the wing.
    """

    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_output(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":"
            + ls_tag
            + ":CD0",
            val=0.0,
        )
