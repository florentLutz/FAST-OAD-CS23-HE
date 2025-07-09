# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .lcc_harness_core_unit_cost import LCCHarnessCoreUnitCost
from .lcc_harness_unit_cost import LCCHarnessUnitCost
from .layer_unit_volume.sizing_conductor_volume_per_length import SizingConductorVolumePerLength
from .layer_unit_volume.sizing_insulation_volume_per_length import SizingInsulationVolumePerLength
from .layer_unit_volume.sizing_shield_volume_per_length import SizingShieldVolumePerLength
from .layer_unit_volume.sizing_sheath_volume_per_length import SizingSheathVolumePerLength


class LCCHarnessCost(om.Group):
    """
    Class that collects all required computations for the DC cable harness cost.
    """

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
            "conductor_layer_volume",
            SizingConductorVolumePerLength(harness_id=harness_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "insulation_layer_volume",
            SizingInsulationVolumePerLength(harness_id=harness_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "shield_layer_volume",
            SizingShieldVolumePerLength(harness_id=harness_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "sheath_layer_volume",
            SizingSheathVolumePerLength(harness_id=harness_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="core_material_cost",
            subsys=LCCHarnessCoreUnitCost(harness_id=harness_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="harness_unit_cost",
            subsys=LCCHarnessUnitCost(harness_id=harness_id),
            promotes=["*"],
        )
