# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .sizing_inductor_energy import SizingInductorEnergy
from .sizing_inductor_iron_surface import SizingInductorIronSurface
from .sizing_inductor_reluctance import SizingInductorReluctance
from .sizing_inductor_turn_number import SizingInductorTurnNumber
from .sizing_inductor_copper_wire_area import SizingInductorCopperWireArea
from .sizing_inductor_core_scaling import SizingInductorCoreScaling
from .sizing_inductor_core_mass import SizingInductorCoreMass
from .sizing_inductor_core_dimensions import SizingInductorCoreDimensions
from .sizing_inductor_copper_mass import SizingInductorCopperMass
from .sizing_inductor_mass import SizingInductorMass
from .sizing_inductor_resistance import SizingInductorResistance

from ..constants import SUBMODEL_CONSTRAINTS_INDUCTOR_AIR_GAP


class SizingInductor(om.Group):
    """
    Class to regroup all the computation related to the sizing of the inductor, to make it easier
    to deactivate for when the component will be taken off of the shelves.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use an inductor",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]

        # It was decided to add the constraints computation at the beginning of the sizing of the
        # inductor instead of along with the other constraints
        option = {"prefix": prefix}
        self.add_subsystem(
            name="constraint_air_gap",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_INDUCTOR_AIR_GAP, options=option
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="inductor_mag_energy",
            subsys=SizingInductorEnergy(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="iron_surface",
            subsys=SizingInductorIronSurface(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_reluctance",
            subsys=SizingInductorReluctance(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="number_of_turns",
            subsys=SizingInductorTurnNumber(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="copper_wire_section_area",
            subsys=SizingInductorCopperWireArea(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_core_scaling",
            subsys=SizingInductorCoreScaling(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="core_mass",
            subsys=SizingInductorCoreMass(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="core_dimensions",
            subsys=SizingInductorCoreDimensions(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="copper_mass",
            subsys=SizingInductorCopperMass(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_mass",
            subsys=SizingInductorMass(prefix=prefix),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_resistance",
            subsys=SizingInductorResistance(prefix=prefix),
            promotes=["*"],
        )
