# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_compressor_pressure_supply import PerformancesCompressorPressureSupply
from .perf_ambient_total_temperature import PerformancesAmbientTotalTemperature
from .perf_ambient_total_pressure import PerformancesAmbientTotalPressure
from .perf_compressor_pressure_target import PerformancesCompressorPressureTarget
from .perf_compressor_power_required import PerformancesCompressorPowerRequired
from .perf_compressor_power_rating import PerformancesCompressorPowerRating

from ...perf_ambient_pressure import PerformancesPEMFCStackBOPAmbientPressure
from ..fluid_characteristics import FluidSpecificHeatCapacity


class PerformancesInlet(om.Group):
    """
    Air inlet Performances group.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="compressor_id",
            default=None,
            desc="Identifier of the compressor",
            allow_none=False,
        )
        self.options.declare(
            name="connected_humidifier_id",
            default=None,
            desc="Identifier of the connected humidifier",
            allow_none=False,
        )
        self.options.declare(
            name="connected_heat_exchanger_id",
            default=None,
            desc="Identifier of the connected heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]
        number_of_points = self.options["number_of_points"]
        connected_humidifier_id = self.options["connected_humidifier_id"]
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]

        self.add_subsystem(
            "compressor_ambient_pressure",
            PerformancesPEMFCStackBOPAmbientPressure(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "compressor_pressure_target",
            PerformancesCompressorPressureTarget(
                pemfc_stack_bop_id=pemfc_stack_bop_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "compressor_pressure_supply",
            PerformancesCompressorPressureSupply(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                connected_humidifier_id=connected_humidifier_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "compressor_ambient_total_pressure",
            PerformancesAmbientTotalPressure(
                number_of_points=number_of_points,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                compressor_id=compressor_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "compressor_ambient_total_temperature",
            PerformancesAmbientTotalTemperature(
                number_of_points=number_of_points,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                compressor_id=compressor_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_specific_heat_capacity",
            FluidSpecificHeatCapacity(
                number_of_points=number_of_points,
                fluid="air",
            ),
            promotes=[
                ("fluid_temperature", "exterior_temperature"),
                ("fluid_pressure", "compressor_pressure_target"),
                ("specific_heat_capacity", "compressed_air_specific_heat_capacity"),
            ],
        )
        self.add_subsystem(
            "compressor_power_required",
            PerformancesCompressorPowerRequired(
                number_of_points=number_of_points,
                compressor_id=compressor_id,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "compressor_power_rating",
            PerformancesCompressorPowerRating(
                number_of_points=number_of_points,
                compressor_id=compressor_id,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
            ),
            promotes=["*"],
        )
