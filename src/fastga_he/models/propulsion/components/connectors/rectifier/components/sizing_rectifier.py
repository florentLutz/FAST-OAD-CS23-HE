# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .sizing_energy_coefficient_scaling import SizingRectifierEnergyCoefficientScaling
from .sizing_energy_coefficients import SizingRectifierEnergyCoefficients
from .sizing_resistance_scaling import SizingRectifierResistanceScaling
from .sizing_reference_resistance import SizingRectifierResistances
from .sizing_thermal_resistance import SizingRectifierThermalResistances
from .sizing_thermal_resistance_casing import SizingRectifierCasingThermalResistance
from .sizing_capacitor_current_caliber import SizingRectifierCapacitorCurrentCaliber
from .sizing_capacitor_capacity import SizingRectifierCapacitorCapacity
from .sizing_capacitor_weight import SizingRectifierCapacitorWeight
from .sizing_dimension_module import SizingRectifierModuleDimension
from .sizing_inductor_current_caliber import SizingRectifierInductorCurrentCaliber
from .sizing_weight_casing import SizingRectifierCasingsWeight
from .sizing_contactor_weight import SizingRectifierContactorWeight
from .sizing_rectifier_cg import SizingRectifierCG
from .sizing_rectifier_drag import SizingRectifierDrag

from .cstr_rectifier import ConstraintsRectifier

from fastga_he.models.propulsion.sub_components.heat_sink.components.sizing_heat_sink import (
    SizingHeatSink,
)
from fastga_he.models.propulsion.sub_components.inductor.components.sizing_inductor import (
    SizingInductor,
)
from fastga_he.powertrain_builder.powertrain import PT_DATA_PREFIX

from ..constants import POSSIBLE_POSITION, SUBMODEL_CONSTRAINTS_RECTIFIER_WEIGHT


class SizingRectifier(om.Group):
    def initialize(self):

        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the rectifier, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_rectifier",
            subsys=ConstraintsRectifier(rectifier_id=rectifier_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="energy_coefficient_scaling",
            subsys=SizingRectifierEnergyCoefficientScaling(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="energy_coefficient",
            subsys=SizingRectifierEnergyCoefficients(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="resistances_scaling",
            subsys=SizingRectifierResistanceScaling(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="reference_resistance",
            subsys=SizingRectifierResistances(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="thermal_resistance",
            subsys=SizingRectifierThermalResistances(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="thermal_resistance_casing",
            subsys=SizingRectifierCasingThermalResistance(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="capacitor_capacity",
            subsys=SizingRectifierCapacitorCapacity(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="capacitor_caliber",
            subsys=SizingRectifierCapacitorCurrentCaliber(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="capacitor_weight",
            subsys=SizingRectifierCapacitorWeight(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="module_dimensions",
            subsys=SizingRectifierModuleDimension(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        rectifier_prefix = PT_DATA_PREFIX + "rectifier:" + rectifier_id
        self.add_subsystem(
            name="heat_sink_sizing",
            subsys=SizingHeatSink(prefix=rectifier_prefix),
            promotes=["*"],
        )

        self.add_subsystem(
            name="inductor_current_caliber",
            subsys=SizingRectifierInductorCurrentCaliber(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_sizing",
            subsys=SizingInductor(prefix=rectifier_prefix),
            promotes=["*"],
        )

        self.add_subsystem(
            name="casings_weight",
            subsys=SizingRectifierCasingsWeight(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="contactors_weight",
            subsys=SizingRectifierContactorWeight(rectifier_id=rectifier_id),
            promotes=["*"],
        )
        rectifier_options = {"rectifier_id": rectifier_id}
        self.add_subsystem(
            name="rectifier_weight",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_RECTIFIER_WEIGHT, options=rectifier_options
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="rectifier_CG",
            subsys=SizingRectifierCG(rectifier_id=rectifier_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:

            system_name = "rectifier_drag_ls" if low_speed_aero else "rectifier_drag_cruise"

            self.add_subsystem(
                name=system_name,
                subsys=SizingRectifierDrag(
                    rectifier_id=rectifier_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
