# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .sizing_pipe_inner_radius import SizingPipeInnerRadius
from .sizing_pipe_wall_thickness import SizingPipeWallThickness
from .sizing_pipe_weight import SizingPipeWeight
from .sizing_pipe_coolant_volume import SizingPipeCoolantVolume

from ..fluid_characteristics import FluidDensity


class SizingPipe(om.Group):
    """
    Sizing of the Pipe in the TMS
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="pipe_id",
            default=None,
            desc="Identifier of the pipe",
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
        fluid = self.options["coolant_fluid_type"]
        pipe_id = self.options["pipe_id"]

        self.add_subsystem(
            "pipe_coolant_density_sizing",
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
            "pipe_inner_radius",
            SizingPipeInnerRadius(pemfc_stack_bop_id=pemfc_stack_bop_id, pipe_id=pipe_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "pipe_wall_thickness",
            SizingPipeWallThickness(pemfc_stack_bop_id=pemfc_stack_bop_id, pipe_id=pipe_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "pipe_total_weight",
            SizingPipeWeight(pemfc_stack_bop_id=pemfc_stack_bop_id, pipe_id=pipe_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "pipe_coolant_volume",
            SizingPipeCoolantVolume(pemfc_stack_bop_id=pemfc_stack_bop_id, pipe_id=pipe_id),
            promotes=["data:*"],
        )

        self.connect(
            "pipe_coolant_density_sizing.fluid_density", "pipe_inner_radius.coolant_density"
        )
