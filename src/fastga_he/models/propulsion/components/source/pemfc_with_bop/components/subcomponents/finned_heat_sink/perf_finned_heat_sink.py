# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om


from ..fluid_characteristics import FluidDensity, FluidDynamicViscosity


class PerformancesFinnedHeatSink(om.Group):
    """
    Finned heat sink performance group.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="finned_heat_sink_id",
            default=None,
            desc="Identifier of the finned heat sink",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]
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
