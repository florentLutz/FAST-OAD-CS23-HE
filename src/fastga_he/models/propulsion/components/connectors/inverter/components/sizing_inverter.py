# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .cstr_inverter import ConstraintsInverter

from .sizing_energy_coefficient_scaling import SizingInverterEnergyCoefficientScaling
from .sizing_energy_coefficients import SizingInverterEnergyCoefficients
from .sizing_resistance_scaling import SizingInverterResistanceScaling
from .sizing_reference_resistance import SizingInverterResistances
from .sizing_thermal_resistance import SizingInverterThermalResistances
from .sizing_thermal_resistance_casing import SizingInverterCasingThermalResistance
from .sizing_weight_casing import SizingInverterCasingsWeight
from .sizing_heat_capacity_casing import SizingInverterCasingHeatCapacity
from .sizing_dimension_module import SizingInverterModuleDimension
from .sizing_heat_sink import SizingHeatSink
from .sizing_capacitor_current_caliber import SizingInverterCapacitorCurrentCaliber
from .sizing_capacitor_capacity import SizingInverterCapacitorCapacity
from .sizing_capacitor_weight import SizingInverterCapacitorWeight
from .sizing_inductor_inductance import SizingInverterInductorInductance
from .sizing_inductor_weight import SizingInverterInductorWeight
from .sizing_contactor_weight import SizingInverterContactorWeight
from .sizing_inverter_weight import SizingInverterWeight
from .sizing_inverter_power_density import SizingInverterPowerDensity


class SizingInverter(om.Group):
    """
    Class that regroups all of the sub components for the computation of the inverter weight.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]

        self.add_subsystem(
            name="constraints_inverter",
            subsys=ConstraintsInverter(inverter_id=inverter_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="energy_coefficient_scaling",
            subsys=SizingInverterEnergyCoefficientScaling(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="energy_coefficient",
            subsys=SizingInverterEnergyCoefficients(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="resistance_scaling",
            subsys=SizingInverterResistanceScaling(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="resistances",
            subsys=SizingInverterResistances(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="thermal_resistances",
            subsys=SizingInverterThermalResistances(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="casing_thermal_resistances",
            subsys=SizingInverterCasingThermalResistance(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="casing_weight",
            subsys=SizingInverterCasingsWeight(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="casing_heat_capacity",
            subsys=SizingInverterCasingHeatCapacity(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="module_dimension",
            subsys=SizingInverterModuleDimension(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="heat_sink_sizing",
            subsys=SizingHeatSink(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="capacitor_current_caliber",
            subsys=SizingInverterCapacitorCurrentCaliber(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="capacitor_capacity",
            subsys=SizingInverterCapacitorCapacity(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="capacitor_weight",
            subsys=SizingInverterCapacitorWeight(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_inductance",
            subsys=SizingInverterInductorInductance(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_weight",
            subsys=SizingInverterInductorWeight(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="contactor_weight",
            subsys=SizingInverterContactorWeight(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inverter_weight",
            subsys=SizingInverterWeight(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inverter_power_density",
            subsys=SizingInverterPowerDensity(inverter_id=inverter_id),
            promotes=["*"],
        )
