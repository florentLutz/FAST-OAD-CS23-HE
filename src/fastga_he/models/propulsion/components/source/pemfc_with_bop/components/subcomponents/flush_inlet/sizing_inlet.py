# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .sizing_throat_height import SizingThroatHeight
from .sizing_inlet_geometry import SizingInletGeometry
from .sizing_inlet_weight import SizingInletWeight


class SizingInlet(om.Group):
    """
    Air flush_inlet sizing computations.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="air_inlet_id",
            default=None,
            desc="Identifier of the air flush_inlet",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_subsystem(
            "throat_height",
            SizingThroatHeight(pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_geometry",
            SizingInletGeometry(pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_weight",
            SizingInletWeight(pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id),
            promotes=["*"],
        )
