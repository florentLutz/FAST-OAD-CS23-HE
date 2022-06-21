# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from .sizing_material_core import MaterialCore
from .sizing_current_per_cable import CurrentPerCable
from .sizing_cable_gauge import CableGauge
from .sizing_resistance_per_length import ResistancePerLength
from .sizing_insulation_thickness import InsulationThickness
from .sizing_mass_per_length import MassPerLength
from .sizing_harness_mass import HarnessMass
from .sizing_reference_resistance import ReferenceResistance
from .sizing_heat_capacity_per_length import HeatCapacityPerLength
from .sizing_heat_capacity import HeatCapacityCable
from .sizing_cable_radius import CableRadius


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

        self.add_subsystem(
            "core_material", MaterialCore(harness_id=harness_id), promotes=["data:*"]
        )
        self.add_subsystem(
            "current_per_cable", CurrentPerCable(harness_id=harness_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "cable_conductor_sizing", CableGauge(harness_id=harness_id), promotes=["data:*"]
        )
        self.add_subsystem(
            "cable_insulation_sizing",
            InsulationThickness(harness_id=harness_id),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            "cable_total_radius",
            CableRadius(harness_id=harness_id),
            promotes=["data:*", "settings:*"],
        )

        self.add_subsystem(
            "resistance_per_length", ResistancePerLength(harness_id=harness_id), promotes=["data:*"]
        )
        self.add_subsystem(
            "mass_per_length",
            MassPerLength(harness_id=harness_id),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            "heat_capacity_per_length",
            HeatCapacityPerLength(harness_id=harness_id),
            promotes=["data:*", "settings:*"],
        )

        self.add_subsystem(
            "resistance", ReferenceResistance(harness_id=harness_id), promotes=["data:*"]
        )
        self.add_subsystem("mass", HarnessMass(harness_id=harness_id), promotes=["data:*"])
        self.add_subsystem(
            "heat_capacity", HeatCapacityCable(harness_id=harness_id), promotes=["data:*"]
        )
