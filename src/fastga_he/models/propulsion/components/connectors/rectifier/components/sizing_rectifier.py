# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_energy_coefficient_scaling import SizingRectifierEnergyCoefficientScaling
from .sizing_energy_coefficients import SizingRectifierEnergyCoefficients
from .sizing_resistance_scaling import SizingRectifierResistanceScaling
from .sizing_reference_resistance import SizingRectifierResistances
from .sizing_thermal_resistance import SizingRectifierThermalResistances
from .sizing_thermal_resistance_casing import SizingRectifierCasingThermalResistance
from .sizing_capacitor_current_caliber import SizingRectifierCapacitorCurrentCaliber
from .sizing_capacitor_capacity import SizingRectifierCapacitorCapacity
from .sizing_capacitor_weight import SizingRectifierCapacitorWeight
from .sizing_rectifier_weight import SizingRectifierWeight
from .sizing_rectifier_cg import SizingRectifierCG
from .sizing_rectifier_drag import SizingRectifierDrag

from .cstr_rectifier import ConstraintsRectifier

from ..constants import POSSIBLE_POSITION


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
            name="rectifier_weight",
            subsys=SizingRectifierWeight(rectifier_id=rectifier_id),
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
