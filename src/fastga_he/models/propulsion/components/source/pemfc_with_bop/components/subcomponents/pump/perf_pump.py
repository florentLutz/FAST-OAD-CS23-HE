# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_pump_volumetric_flow_rate import PerformancesPumpVolumetricFlowRate
from .perf_pressure_drop_sum import PerformancesCoolantSystemPressureDrop
from .perf_pump_required_power import PerformancesPumpPower
from ..fluid_characteristics import FluidDensity


class PerformancesPump(om.Group):
    """
    Pump performance computation.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="pump_id",
            default=None,
            desc="Identifier of the pump",
            allow_none=False,
        )
        self.options.declare(
            "coolant_fluid_type",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )
        self.options.declare(
            name="coolant_component_names",
            default="None",
            desc="A list of the TBS components that use coolant",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        fluid = self.options["coolant_fluid_type"]
        pump_id = self.options["pump_id"]
        coolant_component_names = self.options["coolant_component_names"]

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
                (
                    "fluid_density",
                    "coolant_density",
                ),
            ],
        )
        self.add_subsystem(
            "pump_volumetric_flow_rate",
            PerformancesPumpVolumetricFlowRate(
                pemfc_stack_bop_id=pemfc_stack_bop_id, pump_id=pump_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "pump_pressure_drop",
            PerformancesCoolantSystemPressureDrop(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                pump_id=pump_id,
                coolant_component_names=coolant_component_names,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "pump_required_power",
            PerformancesPumpPower(pemfc_stack_bop_id=pemfc_stack_bop_id, pump_id=pump_id),
            promotes=["*"],
        )
