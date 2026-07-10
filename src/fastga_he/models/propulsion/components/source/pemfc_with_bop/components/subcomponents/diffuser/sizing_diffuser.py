# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .sizing_diffuser_angles import SizingDiffuserAngles
from .sizing_outer_dimension import SizingOuterDimension
from .sizing_inner_volume import SizingInnerVolume
from .sizing_outer_volume import SizingOuterVolume
from .sizing_diffuser_weight import SizingDiffuserWeight
from .sizing_area_ratio import SizingAreaRatio
from .sizing_diffuser_stall_check_ratios import SizingDiffuserStallCheckRatios
from .sizing_cross_section_area import SizingCrossSectionArea
from .sizing_entry_hydraulic_diameter import SizingEntryHydraulicDiameter
from .sizing_diffuser_relative_roughness import SizingDiffuserRelativeRoughness
from .sizing_width_inlet_side import SizingInletSideInnerWidth


class SizingDiffuser(om.Group):
    """
    Diffuser sizing computations.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="diffuser_id",
            default=None,
            desc="Identifier of the diffuser",
            allow_none=False,
        )
        self.options.declare(
            name="connected_heat_exchanger_id",
            default=None,
            desc="Identifier of the connected heat exchanger",
            allow_none=False,
        )
        self.options.declare(
            name="connected_air_inlet_id",
            default=None,
            desc="Identifier of the connected air flush_inlet",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]
        heat_exchanger_id = self.options["connected_heat_exchanger_id"]
        air_inlet_id = self.options["connected_air_inlet_id"]

        self.add_subsystem(
            "sizing_inlet_side_inner_width",
            SizingInletSideInnerWidth(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                connected_air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_diffuser_angles",
            SizingDiffuserAngles(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                connected_heat_exchanger_id=heat_exchanger_id,
                connected_air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_outer_dimension",
            SizingOuterDimension(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                connected_heat_exchanger_id=heat_exchanger_id,
                connected_air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_inner_volume",
            SizingInnerVolume(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                connected_heat_exchanger_id=heat_exchanger_id,
                connected_air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_outer_volume",
            SizingOuterVolume(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_diffuser_weight",
            SizingDiffuserWeight(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_cross_section_area",
            SizingCrossSectionArea(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                connected_heat_exchanger_id=heat_exchanger_id,
                connected_air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_area_ratio",
            SizingAreaRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "stall_check_ratios",
            SizingDiffuserStallCheckRatios(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                connected_air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_entry_hydraulic_diameter",
            SizingEntryHydraulicDiameter(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                connected_air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_diffuser_relative_roughness",
            SizingDiffuserRelativeRoughness(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
            ),
            promotes=["*"],
        )
