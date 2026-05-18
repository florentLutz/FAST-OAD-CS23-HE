# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .sizing_heat_sink_fin_thickness import SizingHeatSinkFinThickness
from .sizing_heat_sink_length import SizingHeatSinkFinLength
from .sizing_heat_sink_fin_height import SizingHeatSinkFinHeightSimplified
from .sizing_heat_sink_weight import SizingHeatSinkMass
from .sizing_heat_sink_added_wet_area import SizingHeatSinkWetArea
from .sizing_heat_sink_fin_spacing import SizingHeatSinkFinSpacing


class SizingFinnedHeatSink(om.Group):
    """
    Sizing of the Pipe in the TMS
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="finned_heat_sink_id",
            default=None,
            desc="Identifier of the finned heat sink",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        self.add_subsystem(
            "sizing_heat_sink_fin_thickness",
            SizingHeatSinkFinThickness(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "sizing_heat_sink_fin_length",
            SizingHeatSinkFinLength(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "sizing_heat_sink_fin_spacing",
            SizingHeatSinkFinSpacing(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "sizing_heat_sink_fin_height",
            SizingHeatSinkFinHeightSimplified(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "sizing_heat_sink_weight",
            SizingHeatSinkMass(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "sizing_heat_sink_added_wet_area",
            SizingHeatSinkWetArea(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
            ),
            promotes=["data:*"],
        )
