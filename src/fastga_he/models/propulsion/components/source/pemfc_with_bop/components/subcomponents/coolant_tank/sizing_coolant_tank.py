# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .sizing_coolant_volume import SizingCoolantTotalVolume
from .sizing_coolant_tank_weight import SizingCoolantTankWeight
from .sizing_coolant_total_weight import SizingCoolantTotalWeight

from ..fluid_characteristics import FluidDensity


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
            name="coolant_component_ids",
            default="None",
            desc="A list of the TBS components that use coolant",
            allow_none=False,
        )
        self.options.declare(
            "coolant_fluid_type",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_tank_id = self.options["coolant_tank_id"]
        coolant_component_ids = self.options["coolant_component_ids"]
        fluid = self.options["coolant_fluid_type"]

        self.add_subsystem(
            "coolant_density_sizing",
            FluidDensity(fluid=fluid),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:static_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:mean_temperature",
                ),
            ],
        )
        self.add_subsystem(
            "coolant_total_volume",
            SizingCoolantTotalVolume(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                coolant_tank_id=coolant_tank_id,
                coolant_component_ids=coolant_component_ids,
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
        self.add_subsystem(
            "total_weight",
            SizingCoolantTotalWeight(
                pemfc_stack_bop_id=pemfc_stack_bop_id, coolant_tank_id=coolant_tank_id
            ),
            promotes=["data:*"],
        )

        self.connect("coolant_density_sizing.fluid_density", "total_weight.coolant_density")
