# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_capacitor_capacity_scaling import SizingCapacitorCapacityScaling
from .sizing_capacitor_diameter_scaling import SizingCapacitorDiameterScaling
from .sizing_capacitor_diameter import SizingCapacitorDiameter
from .sizing_capacitor_height import SizingCapacitorHeight
from .sizing_capacitor_height_scaling import SizingCapacitorHeightScaling
from .sizing_capacitor_thermal_resistance_scaling import (
    SizingCapacitorThermalResistanceScaling,
)
from .sizing_capacitor_thermal_resistance import SizingCapacitorThermalResistance
from .sizing_capacitor_resistance_scaling import SizingCapacitorResistanceScaling
from .sizing_capacitor_resistance import SizingCapacitorResistance
from .sizing_capacitor_mass_scaling import SizingCapacitorMassScaling
from .sizing_capacitor_mass import SizingCapacitorMass


class SizingCapacitor(om.Group):
    """
    Class to regroup all the computation related to the sizing of the capacitor, to make it easier
    to deactivate for when the component will be taken off of the shelves.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use a capacitor",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_subsystem(
            name="capacity_scaling",
            subsys=SizingCapacitorCapacityScaling(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="diameter_scaling",
            subsys=SizingCapacitorDiameterScaling(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="diameter",
            subsys=SizingCapacitorDiameter(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="height",
            subsys=SizingCapacitorHeight(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="height_scaling",
            subsys=SizingCapacitorHeightScaling(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="thermal_resistance_scaling",
            subsys=SizingCapacitorThermalResistanceScaling(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="thermal_resistance",
            subsys=SizingCapacitorThermalResistance(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="resistance_scaling",
            subsys=SizingCapacitorResistanceScaling(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="resistance",
            subsys=SizingCapacitorResistance(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="mass_scaling",
            subsys=SizingCapacitorMassScaling(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="mass",
            subsys=SizingCapacitorMass(prefix=prefix),
            promotes=["*"],
        )
