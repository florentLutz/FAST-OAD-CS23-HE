# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_DC_LINE

from .sizing_material_core import SizingMaterialCore
from .sizing_current_per_cable import SizingCurrentPerCable
from .sizing_cable_gauge import SizingCableGauge
from .sizing_resistance_per_length import SizingResistancePerLength
from .sizing_insulation_thickness import SizingInsulationThickness
from .sizing_sheath_thickness import SizingCableSheathThickness
from .sizing_mass_per_length import SizingMassPerLength
from .sizing_contactor_mass import SizingHarnessContactorMass
from .sizing_harness_mass import SizingHarnessMass
from .sizing_reference_resistance import SizingReferenceResistance
from .sizing_heat_capacity_per_length import SizingHeatCapacityPerLength
from .sizing_heat_capacity import SizingHeatCapacityCable
from .sizing_cable_radius import SizingCableRadius


class SizingHarness(om.Group):
    def initialize(self):
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        option_harness_id = {"harness_id": harness_id}

        self.add_subsystem(
            name="constraints_dc_line",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_LINE, options=option_harness_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "core_material", SizingMaterialCore(harness_id=harness_id), promotes=["data:*"]
        )
        self.add_subsystem(
            "current_per_cable", SizingCurrentPerCable(harness_id=harness_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "cable_conductor_sizing", SizingCableGauge(harness_id=harness_id), promotes=["data:*"]
        )
        self.add_subsystem(
            "cable_insulation_sizing",
            SizingInsulationThickness(harness_id=harness_id),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            "cable_sheath_sizing",
            SizingCableSheathThickness(harness_id=harness_id),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            "cable_total_radius",
            SizingCableRadius(harness_id=harness_id),
            promotes=["data:*", "settings:*"],
        )

        self.add_subsystem(
            "resistance_per_length",
            SizingResistancePerLength(harness_id=harness_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "mass_per_length",
            SizingMassPerLength(harness_id=harness_id),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            "contactor_mass",
            SizingHarnessContactorMass(harness_id=harness_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "heat_capacity_per_length",
            SizingHeatCapacityPerLength(harness_id=harness_id),
            promotes=["data:*", "settings:*"],
        )

        self.add_subsystem(
            "resistance", SizingReferenceResistance(harness_id=harness_id), promotes=["data:*"]
        )
        self.add_subsystem("mass", SizingHarnessMass(harness_id=harness_id), promotes=["data:*"])
        self.add_subsystem(
            "heat_capacity", SizingHeatCapacityCable(harness_id=harness_id), promotes=["data:*"]
        )
