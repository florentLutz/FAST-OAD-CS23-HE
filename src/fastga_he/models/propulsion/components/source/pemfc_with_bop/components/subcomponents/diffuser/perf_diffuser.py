# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_diffuser_exit_air_speed import PerformancesDiffuserExitAirSpeed
from .perf_diffuser_average_air_speed import PerformancesDiffuserAverageAirSpeed
from .perf_diffuser_first_stall_angles import PerformancesDiffuserFirstStallAngel
from .perf_diffuser_appreciable_stall_angles import PerformancesDiffuserAppreciableStallAngles
from .perf_diffuser_reynolds_number import PerformancesDiffuserReynoldsNumber
from .perf_diffuser_darcy_friction_factor import PerformancesDiffuserDarcyFrictionFactor
from .perf_diffuser_singular_pressure_loss_coeff import (
    PerformancesDiffuserSingularPressureLossCoefficient,
)
from .perf_diffuser_expansion_loss_coeff import PerformancesDiffuserExpansionLossCoefficient
from .perf_diffuser_friction_loss_coeff import PerformancesDiffuserFrictionLossCoefficient
from .perf_diffuser_pressure_drop import PerformancesDiffuserPressureDrop
from .perf_diffuser_exit_pressure import PerformancesDiffuserExitPressure
from .perf_diffuser_exit_temperature import PerformancesDiffuserExitTemperature
from .perf_diffuser_exit_total_pressure import PerformancesDiffuserExitTotalPressure

from ..fluid_characteristics import FluidDensity, FluidDynamicViscosity, FluidSpecificHeatCapacity


class PerformancesDiffuser(om.Group):
    """
    Diffuser performance computations.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="diffuser_id",
            default=None,
            desc="Identifier of the diffuser",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "perf_diffuser_exit_air_speed",
            PerformancesDiffuserExitAirSpeed(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_average_air_speed",
            PerformancesDiffuserAverageAirSpeed(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_first_stall_angles",
            PerformancesDiffuserFirstStallAngel(
                pemfc_stack_bop_id=pemfc_stack_bop_id, diffuser_id=diffuser_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_appreciable_stall_angles",
            PerformancesDiffuserAppreciableStallAngles(
                pemfc_stack_bop_id=pemfc_stack_bop_id, diffuser_id=diffuser_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "diffuser_air_density",
            FluidDensity(number_of_points=number_of_points),
            promotes=[
                ("fluid_temperature", "exterior_temperature"),
                ("fluid_pressure", "ambient_pressure"),
                ("fluid_density", "diffuser_air_density"),
            ],
        )
        self.add_subsystem(
            "diffuser_air_dynamic_viscosity",
            FluidDynamicViscosity(number_of_points=number_of_points),
            promotes=[
                ("fluid_temperature", "exterior_temperature"),
                ("fluid_pressure", "ambient_pressure"),
                ("fluid_dynamic_viscosity", "diffuser_air_dynamic_viscosity"),
            ],
        )
        self.add_subsystem(
            "diffuser_air_specific_heat_capacity",
            FluidSpecificHeatCapacity(number_of_points=number_of_points),
            promotes=[
                ("fluid_temperature", "exterior_temperature"),
                ("fluid_pressure", "ambient_pressure"),
                ("fluid_specific_heat_capacity", "diffuser_air_specific_heat_capacity"),
            ],
        )
        self.add_subsystem(
            "perf_diffuser_reynolds_number",
            PerformancesDiffuserReynoldsNumber(
                number_of_points=number_of_points,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_darcy_friction_factor",
            PerformancesDiffuserDarcyFrictionFactor(
                number_of_points=number_of_points,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_singular_pressure_loss_coeff",
            PerformancesDiffuserSingularPressureLossCoefficient(
                pemfc_stack_bop_id=pemfc_stack_bop_id, diffuser_id=diffuser_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_expansion_loss_coeff",
            PerformancesDiffuserExpansionLossCoefficient(
                pemfc_stack_bop_id=pemfc_stack_bop_id, diffuser_id=diffuser_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_friction_loss_coeff",
            PerformancesDiffuserFrictionLossCoefficient(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_pressure_drop",
            PerformancesDiffuserPressureDrop(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_exit_pressure",
            PerformancesDiffuserExitPressure(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_exit_temperature",
            PerformancesDiffuserExitTemperature(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "perf_diffuser_exit_total_pressure",
            PerformancesDiffuserExitTotalPressure(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
