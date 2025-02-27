# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..components.perf_direct_bus_connection import PerformancesPEMFCStackDirectBusConnection
from ..components.perf_pemfc_power import PerformancesPEMFCStackPower
from ..components.perf_maximum import PerformancesPEMFCStackMaximum
from ..components.perf_pemfc_current_density import PerformancesPEMFCStackCurrentDensity
from ..components.perf_fuel_consumption import PerformancesPEMFCStackFuelConsumption
from ..components.perf_fuel_consumed import PerformancesPEMFCStackFuelConsumed
from ..components.perf_pemfc_efficiency import PerformancesPEMFCStackEfficiency
from ..components.perf_pemfc_voltage import PerformancesPEMFCStackVoltage
from ..components.perf_pemfc_layer_voltage import (
    PerformancesPEMFCStackSingleLayerVoltageEmpirical,
    PerformancesPEMFCStackSingleLayerVoltageAnalytical,
)


class PerformancesPEMFCStack(om.Group):
    """Class that regroups all the subcomponents for the PEMFC stack performance computation."""

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
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

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]
        compressor_connection = self.options["compressor_connection"]
        direct_bus_connection = self.options["direct_bus_connection"]
        model_fidelity = self.options["model_fidelity"]

        self.add_subsystem(
            "pemfc_current_density",
            PerformancesPEMFCStackCurrentDensity(
                pemfc_stack_id=pemfc_stack_id,
                number_of_points=number_of_points,
                model_fidelity=model_fidelity,
            ),
            promotes=["*"],
        )

        if model_fidelity == "analytical":
            self.add_subsystem(
                "pemfc_layer_voltage",
                PerformancesPEMFCStackSingleLayerVoltageAnalytical(
                    pemfc_stack_id=pemfc_stack_id,
                    number_of_points=number_of_points,
                    compressor_connection=compressor_connection,
                ),
                promotes=["*"],
            )

        else:
            self.add_subsystem(
                "pemfc_layer_voltage",
                PerformancesPEMFCStackSingleLayerVoltageEmpirical(
                    pemfc_stack_id=pemfc_stack_id,
                    number_of_points=number_of_points,
                    compressor_connection=compressor_connection,
                ),
                promotes=["*"],
            )

        self.add_subsystem(
            "pemfc_voltage",
            PerformancesPEMFCStackVoltage(
                pemfc_stack_id=pemfc_stack_id,
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
            PerformancesPEMFCStackFuelConsumption(
                pemfc_stack_id=pemfc_stack_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "fuel_consumed",
            PerformancesPEMFCStackFuelConsumed(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "pemfc_efficiency",
            PerformancesPEMFCStackEfficiency(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_power",
            PerformancesPEMFCStackPower(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "maximum",
            PerformancesPEMFCStackMaximum(
                pemfc_stack_id=pemfc_stack_id, number_of_points=number_of_points
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
