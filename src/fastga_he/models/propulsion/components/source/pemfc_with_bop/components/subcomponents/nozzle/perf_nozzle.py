# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_nozzle_air_speed import PerformancesNozzleAirSpeed
from .perf_nozzle_inlet_pressure import PerformancesNozzleInletPressure
from .perf_nozzle_inlet_temperature import PerformancesNozzleInletTemperature
from .perf_nozzle_reynolds_number import PerformancesNozzleReynoldsNumber
from .perf_nozzle_darcy_friction_factor import PerformancesNozzleDarcyFrictionFactor
from .perf_nozzle_contraction_loss_coeff import PerformancesNozzleContractionLossCoefficient
from .perf_nozzle_friction_loss_coeff import PerformancesNozzleFrictionLossCoefficient
from .perf_nozzle_pressure_drop import PerformancesNozzlePressureDrop
from .perf_nozzle_exit_pressure import PerformancesNozzleExitPressure
from .perf_nozzle_drag import PerformancesNozzleDrag

from ..fluid_characteristics import FluidDensity, FluidDynamicViscosity, FluidSpecificHeatCapacity


class PerformancesNozzle(om.Group):
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
            name="nozzle_id",
            default=None,
            desc="Identifier of the connected nozzle",
            allow_none=False,
        )
        self.options.declare(
            name="connected_heat_exchanger_id",
            default=None,
            desc="Identifier of the connected heat exchanger",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]
        number_of_points = self.options["number_of_points"]
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]

        self.add_subsystem(
            "nozzle_air_speed",
            PerformancesNozzleAirSpeed(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "nozzle_inlet_pressure",
            PerformancesNozzleInletPressure(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "nozzle_inlet_temperature",
            PerformancesNozzleInletTemperature(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "nozzle_air_density",
            FluidDensity(number_of_points=number_of_points),
            promotes=[
                ("fluid_temperature", "nozzle_inlet_temperature"),
                ("fluid_pressure", "nozzle_inlet_pressure"),
                ("fluid_density", "nozzle_air_density"),
            ],
        )
        self.add_subsystem(
            "nozzle_air_dynamic_viscosity",
            FluidDynamicViscosity(number_of_points=number_of_points),
            promotes=[
                ("fluid_temperature", "nozzle_inlet_temperature"),
                ("fluid_pressure", "nozzle_inlet_pressure"),
                ("fluid_dynamic_viscosity", "nozzle_air_dynamic_viscosity"),
            ],
        )
        self.add_subsystem(
            "nozzle_air_specific_heat_capacity",
            FluidSpecificHeatCapacity(number_of_points=number_of_points),
            promotes=[
                ("fluid_temperature", "nozzle_inlet_temperature"),
                ("fluid_pressure", "nozzle_inlet_pressure"),
                ("fluid_specific_heat_capacity", "nozzle_air_specific_heat_capacity"),
            ],
        )
        self.add_subsystem(
            "nozzle_reynolds_number",
            PerformancesNozzleReynoldsNumber(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "nozzle_darcy_friction_factor",
            PerformancesNozzleDarcyFrictionFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "nozzle_contraction_loss_coeff",
            PerformancesNozzleContractionLossCoefficient(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "nozzle_friction_loss_coeff",
            PerformancesNozzleFrictionLossCoefficient(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "nozzle_pressure_drop",
            PerformancesNozzlePressureDrop(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "nozzle_exit_pressure",
            PerformancesNozzleExitPressure(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "nozzle_drag",
            PerformancesNozzleDrag(
                number_of_points=number_of_points,
                nozzle_id=nozzle_id,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
            ),
            promotes=["*"],
        )
