# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

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
from .sizing_capacitor_current_caliber import SizingInverterCapacitorCurrentCaliber
from .sizing_capacitor_capacity import SizingInverterCapacitorCapacity
from .sizing_inductor_current_caliber import SizingInverterInductorCurrentCaliber
from .sizing_contactor_weight import SizingInverterContactorWeight
from .sizing_inverter_weight import SizingInverterWeight
from .sizing_inverter_power_density import SizingInverterPowerDensity
from .sizing_inverter_cg import SizingInverterCGX
from .sizing_inverter_drag import SizingInverterDrag

from fastga_he.models.propulsion.sub_components import (
    SizingHeatSink,
    SizingCapacitor,
    SizingInductor,
)
from fastga_he.powertrain_builder.powertrain import PT_DATA_PREFIX

from ..constants import POSSIBLE_POSITION


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
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the inverter, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]
        position = self.options["position"]

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

        # The number of modules  in the rectifier isn't really up to the user to change,
        # it depends on the topology. Here, the topology requires 3 modules to be cooled by one
        # heat sink
        ivc_module_number = om.IndepVarComp()
        ivc_module_number.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":module:number",
            val=3,
        )
        self.add_subsystem(name="module_number", subsys=ivc_module_number, promotes=["*"])

        inverter_prefix = PT_DATA_PREFIX + "inverter:" + inverter_id
        self.add_subsystem(
            name="heat_sink_sizing",
            subsys=SizingHeatSink(prefix=inverter_prefix),
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
            subsys=SizingCapacitor(prefix=inverter_prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_current_caliber",
            subsys=SizingInverterInductorCurrentCaliber(inverter_id=inverter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_weight",
            subsys=SizingInductor(prefix=inverter_prefix),
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
        self.add_subsystem(
            name="inverter_CG",
            subsys=SizingInverterCGX(inverter_id=inverter_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "inverter_drag_ls" if low_speed_aero else "inverter_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingInverterDrag(
                    inverter_id=inverter_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
