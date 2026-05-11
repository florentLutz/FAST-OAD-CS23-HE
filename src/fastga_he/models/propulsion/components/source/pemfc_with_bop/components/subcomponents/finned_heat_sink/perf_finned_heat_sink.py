# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_heat_sink_base_temperature import PerformancesHeatSinkBasedTemperature
from .perf_base_temperature_difference import PerformancesFinBaseTemperatureDifference
from .perf_air_reynold_number import PerformancesAirReynoldsNumber
from .perf_air_nusselt_number import PerformancesAirNusseltNumber
from .perf_air_thermal_conductivity import PerformancesAirThermalConductivity
from .perf_air_convection_heat_transfer_coefficient import (
    PerformancesAirConvectionHeatTransferCoefficient,
)
from .perf_fin_parameter import PerformancesFinParameter
from .perf_fin_heat_transfer_parameter import PerformancesFinHeatTransferParameter
from .perf_design_dissipation_power import PerformancesDesignDissipationPower
from ...perf_ambient_pressure import PerformancesPEMFCStackBOPAmbientPressure
from ......loads.sm_pmsm.components.perf_air_dynamic_viscosity import (
    PerformancesAirDynamicViscosity,
)

from ..fluid_characteristics import FluidPrandtlNumber, FluidThermalConductivity


class PerformancesFinnedHeatSink(om.Group):
    """
    Finned heat sink performance group.
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
            name="finned_heat_sink_id",
            default=None,
            desc="Identifier of the finned heat sink",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        self.add_subsystem(
            "heat_sink_base_temperature",
            PerformancesHeatSinkBasedTemperature(
                pemfc_stack_bop_id=pemfc_stack_bop_id, finned_heat_sink_id=finned_heat_sink_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "temperature_difference",
            PerformancesFinBaseTemperatureDifference(
                number_of_points=number_of_points,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_dynamic_viscosity",
            PerformancesAirDynamicViscosity(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_reynolds_number",
            PerformancesAirReynoldsNumber(
                number_of_points=number_of_points,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "ambient_pressure",
            PerformancesPEMFCStackBOPAmbientPressure(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_prandtl_number",
            FluidPrandtlNumber(
                number_of_points=number_of_points,
            ),
            promotes=[
                ("fluid_temperature", "exterior_temperature"),
                ("fluid_pressure", "ambient_pressure"),
                ("fluid_prandtl_number", "prandtl_number"),
            ],
        )
        self.add_subsystem(
            "air_nusselt_number",
            PerformancesAirNusseltNumber(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_thermal_conductivity",
            FluidThermalConductivity(
                number_of_points=number_of_points,
            ),
            promotes=[
                ("fluid_temperature", "exterior_temperature"),
                ("fluid_pressure", "ambient_pressure"),
                ("fluid_thermal_conductivity", "air_thermal_conductivity"),
            ],
        )
        self.add_subsystem(
            "design_air_thermal_conductivity",
            PerformancesAirThermalConductivity(
                number_of_points=number_of_points,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "convection_heat_transfer_coefficient",
            PerformancesAirConvectionHeatTransferCoefficient(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "fin_parameter",
            PerformancesFinParameter(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "fin_heat_transfer_parameter",
            PerformancesFinHeatTransferParameter(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "design_dissipation_power",
            PerformancesDesignDissipationPower(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["*"],
        )
