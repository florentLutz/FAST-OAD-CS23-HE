# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_pipe_reynolds_number import PerformancesPipeReynoldsNumber
from .perf_pipe_darcy_friction_factor import PerformancesPipeDarcyFrictionFactor
from .perf_pipe_rating_pressure_drop import PerformancesPipeRatingPressureDrop

from ..fluid_characteristics import FluidDensity, FluidDynamicViscosity


class PerformancesPipe(om.Group):
    """
    Pipe pressure drop computations.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
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

        self.add_subsystem(
            "pipe_coolant_density_performance",
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
            "pipe_coolant_dynamic_viscosity_performance",
            FluidDynamicViscosity(fluid=fluid),
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
            "pipe_reynolds_number",
            PerformancesPipeReynoldsNumber(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "pipe_darcy_friction_factor",
            PerformancesPipeDarcyFrictionFactor(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "pipe_rating_pressure_drop",
            PerformancesPipeRatingPressureDrop(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )

        self.connect(
            "pipe_coolant_density_performance:fluid_density", "pipe_reynolds_number:coolant_density"
        )
        self.connect(
            "pipe_coolant_dynamic_viscosity_performance.fluid_dynamic_viscosity",
            "pipe_reynolds_number.coolant_dynamic_viscosity",
        )
        self.connect(
            "pipe_coolant_density_performance:fluid_density",
            "pipe_rating_pressure_drop:coolant_density",
        )
