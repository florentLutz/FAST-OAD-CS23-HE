# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..components.perf_direct_bus_connection import PerformancesPEMFCStackDirectBusConnection
from ..components.perf_pemfc_power import PerformancesPEMFCStackBOPPower
from ..components.perf_pemfc_thermal_power import PerformancesPEMFCStackBOPThermalPower
from ..components.perf_pemfc_coolant_temperature import PerformancesPEMFCStackBOPCoolantTemperature
from ..components.perf_maximum import PerformancesPEMFCStackBOPMaximum
from ..components.perf_pemfc_current_density import PerformancesPEMFCStackBOPCurrentDensity
from ..components.perf_fuel_consumption import PerformancesPEMFCStackBOPFuelConsumption
from ..components.perf_ambient_air_consumption import PerformancesPEMFCStackBOPAirConsumption
from ..components.perf_fuel_consumed import PerformancesPEMFCStackBOPFuelConsumed
from ..components.perf_pemfc_efficiency import PerformancesPEMFCStackBOPEfficiency
from ..components.perf_pemfc_voltage import PerformancesPEMFCStackBOPVoltage
from ..components.perf_pemfc_layer_voltage import (
    PerformancesPEMFCStackBOPSingleLayerVoltageEmpirical,
    PerformancesPEMFCStackBOPSingleLayerVoltageAnalytical,
)
from .perf_pemfc_bop_current_supply import PerformancesPEMFCStackBOPCurrentSupply
from .subcomponents.perf_pemfc_bop import PerformancesPEMFCBOP


class PerformancesPEMFCStackBOP(om.Group):
    """Class that regroups all the subcomponents for the PEMFC stack performance computations."""

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="direct_bus_connection",
            default=False,
            types=bool,
            desc="If the PEMFC stack is directly connected to a bus, a special mode is required to "
            "interface the two",
        )
        self.options.declare(
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC stack operation pressure have to adjust based on compressor "
            "connection for the oxygen/air inlet",
        )
        self.options.declare(
            name="model_fidelity",
            default="empirical",
            desc="Select the polarization model between empirical and analytical. The "
            "Aerostak 200W empirical polarization model is set as default.",
        )
        self.options.declare(
            "coolant_fluid_type",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_connection = self.options["compressor_connection"]
        direct_bus_connection = self.options["direct_bus_connection"]
        model_fidelity = self.options["model_fidelity"]
        coolant_fluid_type = self.options["coolant_fluid_type"]

        self.add_subsystem(
            "pemfc_current_density",
            PerformancesPEMFCStackBOPCurrentDensity(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                model_fidelity=model_fidelity,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "coolant_temperature",
            PerformancesPEMFCStackBOPCoolantTemperature(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )

        if model_fidelity == "analytical":
            self.add_subsystem(
                "pemfc_layer_voltage",
                PerformancesPEMFCStackBOPSingleLayerVoltageAnalytical(
                    pemfc_stack_bop_id=pemfc_stack_bop_id,
                    number_of_points=number_of_points,
                    compressor_connection=compressor_connection,
                ),
                promotes=["*"],
            )

        else:
            self.add_subsystem(
                "pemfc_layer_voltage",
                PerformancesPEMFCStackBOPSingleLayerVoltageEmpirical(
                    pemfc_stack_bop_id=pemfc_stack_bop_id,
                    number_of_points=number_of_points,
                    compressor_connection=compressor_connection,
                ),
                promotes=["*"],
            )

        self.add_subsystem(
            "pemfc_voltage",
            PerformancesPEMFCStackBOPVoltage(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                direct_bus_connection=direct_bus_connection,
            ),
            promotes=["*"],
        )

        if self.options["direct_bus_connection"]:
            self.add_subsystem(
                "direct_bus_connection",
                PerformancesPEMFCStackDirectBusConnection(number_of_points=number_of_points),
                promotes=["*"],
            )

        self.add_subsystem(
            "fuel_consumption",
            PerformancesPEMFCStackBOPFuelConsumption(
                pemfc_stack_bop_id=pemfc_stack_bop_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_consumption",
            PerformancesPEMFCStackBOPAirConsumption(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "fuel_consumed",
            PerformancesPEMFCStackBOPFuelConsumed(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "pemfc_efficiency",
            PerformancesPEMFCStackBOPEfficiency(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_power",
            PerformancesPEMFCStackBOPPower(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_total_thermal_power",
            PerformancesPEMFCStackBOPThermalPower(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "maximum",
            PerformancesPEMFCStackBOPMaximum(
                pemfc_stack_bop_id=pemfc_stack_bop_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )

        energy_consumed = om.IndepVarComp()
        energy_consumed.add_output(
            "non_consumable_energy_t", np.full(number_of_points, 0.0), units="W*h"
        )
        self.add_subsystem(
            "energy_consumed",
            energy_consumed,
            promotes=["non_consumable_energy_t"],
        )

        if compressor_connection:
            self.add_subsystem(
                "pemfc_bop",
                PerformancesPEMFCBOP(
                    pemfc_stack_bop_id=pemfc_stack_bop_id,
                    number_of_points=number_of_points,
                    coolant_fluid_type=coolant_fluid_type,
                    compressor_id="compressor_1",
                    pipe_id="pipe_1",
                    air_inlet_id="air_inlet_1",
                    primary_heat_exchanger_id="primary_heat_exchanger_1",
                    supplement_heat_exchanger_id="supplement_heat_exchanger_1",
                    humidifier_id="humidifier_1",
                    diffuser_id="diffuser_1",
                    nozzle_id="nozzle_1",
                    pump_id="pump_1",
                ),
                promotes=["*"],
            )

        self.add_subsystem(
            "supply_current",
            PerformancesPEMFCStackBOPCurrentSupply(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.DirectSolver()
