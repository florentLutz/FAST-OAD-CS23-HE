# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..components.perf_direct_bus_connection import PerformancesPEMFCStackDirectBusConnection
from ..components.perf_pemfc_power import PerformancesPEMFCStackPower
from ..components.perf_fuel_power_density import PerformancesPEMFCStackHydrogenPowerDensity
from ..components.perf_maximum import PerformancesPEMFCStackMaximum
from ..components.perf_pemfc_current_density import PerformancesPEMFCStackCurrentDensity

from ..components.perf_fuel_consumption import PerformancesPEMFCStackFuelConsumption
from ..components.perf_fuel_consumed import PerformancesPEMFCStackFuelConsumed
from ..components.perf_pemfc_efficiency import PerformancesPEMFCStackEfficiency
from ..components.perf_pemfc_voltage import PerformancesPEMFCStackVoltage
from ..components.perf_pemfc_expect_specific_power import (
    PerformancesPEMFCStackExpectedSpecificPower,
)
from ..components.perf_pemfc_expect_power_density import PerformancesPEMFCStackExpectedPowerDensity


from ..constants import SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE


class PerformancesPEMFCStack(om.Group):
    """Class that regroups all the subcomponents for PEMFC stack performance computation."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )
        self.options.declare(
            name="direct_bus_connection",
            default=False,
            types=bool,
            desc="If the battery is directly connected to a bus, a special mode is required to "
            "interface the two",
        )
        self.options.declare(
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC operation pressure have to adjust based on compressor connection for "
            "oxygen inlet",
        )
        self.options.declare(
            "max_current_density",
            default=0.7,
            desc="maximum current density of pemfc [A/cm**2]",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]
        compressor_connection = self.options["compressor_connection"]
        direct_bus_connection = self.options["direct_bus_connection"]
        max_current_density = self.options["max_current_density"]
        option_layer_voltage = {
            "number_of_points": number_of_points,
            "pemfc_stack_id": pemfc_stack_id,
            "max_current_density": max_current_density,
            "compressor_connection": compressor_connection,
        }

        self.add_subsystem(
            "pemfc_current_density",
            PerformancesPEMFCStackCurrentDensity(
                number_of_points=number_of_points,
                pemfc_stack_id=pemfc_stack_id,
                max_current_density=max_current_density,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="pemfc_layer_voltage",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE, options=option_layer_voltage
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_voltage",
            PerformancesPEMFCStackVoltage(
                number_of_points=number_of_points,
                direct_bus_connection=direct_bus_connection,
                pemfc_stack_id=pemfc_stack_id,
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
                number_of_points=number_of_points, pemfc_stack_id=pemfc_stack_id
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
                number_of_points=number_of_points, pemfc_stack_id=pemfc_stack_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "hydrogen_power_density",
            PerformancesPEMFCStackHydrogenPowerDensity(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_specific_power",
            PerformancesPEMFCStackExpectedSpecificPower(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_power_density",
            PerformancesPEMFCStackExpectedPowerDensity(pemfc_stack_id=pemfc_stack_id),
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
