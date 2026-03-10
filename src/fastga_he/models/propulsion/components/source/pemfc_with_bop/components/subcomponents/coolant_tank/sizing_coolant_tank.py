# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .sizing_coolant_volume import SizingCoolantTotalVolume
from .sizing_coolant_tank_weight import SizingCoolantTankWeight


class SizingCoolantTank(om.Group):
    """
    Sizing of the TMS coolant tank
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="coolant_tank_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )
        self.options.declare(
            name="coolant_component_names",
            default="None",
            desc="A list of the TBS components that use coolant",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_tank_id = self.options["coolant_tank_id"]
        coolant_component_names = self.options["coolant_component_names"]

        self.add_subsystem(
            "coolant_total_volume",
            SizingCoolantTotalVolume(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                coolant_tank_id=coolant_tank_id,
                coolant_component_names=coolant_component_names,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "tank_weight",
            SizingCoolantTankWeight(
                pemfc_stack_bop_id=pemfc_stack_bop_id, coolant_tank_id=coolant_tank_id
            ),
            promotes=["data:*"],
        )
