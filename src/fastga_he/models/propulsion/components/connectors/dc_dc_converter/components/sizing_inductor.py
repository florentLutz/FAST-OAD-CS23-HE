# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .sizing_inductor_inductance import SizingDCDCConverterInductorInductance
from .sizing_inductor_energy import SizingDCDCConverterInductorEnergy
from .sizing_inductor_iron_surface import SizingDCDCConverterInductorIronSurface
from .sizing_inductor_reluctance import SizingDCDCConverterInductorReluctance
from .sizing_inductor_turn_number import SizingDCDCConverterInductorTurnNumber
from .sizing_inductor_copper_wire_area import SizingDCDCConverterInductorCopperWireArea
from .sizing_inductor_core_scaling import SizingDCDCConverterInductorCoreScaling
from .sizing_inductor_core_mass import SizingDCDCConverterInductorCoreMass
from .sizing_inductor_core_dimensions import SizingDCDCConverterInductorCoreDimensions
from .sizing_inductor_copper_mass import SizingDCDCConverterInductorCopperMass
from .sizing_inductor_mass import SizingDCDCConverterInductorMass
from .sizing_inductor_resistance import SizingDCDCConverterInductorResistance

from ..constants import SUBMODEL_CONSTRAINTS_DC_DC_INDUCTOR_AIR_GAP


class SizingDCDCConverterInductor(om.Group):
    """
    Class to regroup all the computation related to the sizing of the inductor, to make it easier
    to deactivate for when the component will be taken off of the shelves.
    """

    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        # It was decided to add the constraints computation at the beginning of the sizing of the
        # inductor instead of along with the other constraints
        option = {"dc_dc_converter_id": dc_dc_converter_id}
        self.add_subsystem(
            name="constraint_air_gap",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_DC_INDUCTOR_AIR_GAP, options=option
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="inductor_inductance",
            subsys=SizingDCDCConverterInductorInductance(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_mag_energy",
            subsys=SizingDCDCConverterInductorEnergy(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="iron_surface",
            subsys=SizingDCDCConverterInductorIronSurface(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_reluctance",
            subsys=SizingDCDCConverterInductorReluctance(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="number_of_turns",
            subsys=SizingDCDCConverterInductorTurnNumber(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="copper_wire_section_area",
            subsys=SizingDCDCConverterInductorCopperWireArea(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_core_scaling",
            subsys=SizingDCDCConverterInductorCoreScaling(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="core_mass",
            subsys=SizingDCDCConverterInductorCoreMass(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="core_dimensions",
            subsys=SizingDCDCConverterInductorCoreDimensions(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="copper_mass",
            subsys=SizingDCDCConverterInductorCopperMass(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_mass",
            subsys=SizingDCDCConverterInductorMass(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="inductor_resistance",
            subsys=SizingDCDCConverterInductorResistance(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
