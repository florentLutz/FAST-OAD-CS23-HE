# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .sizing_nozzle_length import SizingNozzleLength
from .sizing_nozzle_exit_area import SizingNozzleExitArea
from .sizing_nozzle_area_ratio import SizingNozzleAreaRatio
from .sizing_nozzle_exit_dimension import SizingNozzleExitDimension
from .sizing_outer_dimension import SizingOuterDimension
from .sizing_inner_volume import SizingInnerVolume
from .sizing_outer_volume import SizingOuterVolume
from .sizing_nozzle_weight import SizingNozzleWeight
from .sizing_entry_hydraulic_diameter import SizingEntryHydraulicDiameter
from .sizing_nozzle_relative_roughness import SizingNozzleRelativeRoughness
from .sizing_nozzle_exit_height_length_ratio import SizingNozzleExitHeightLengthRatio
from .sizing_nozzle_alpha_angle import SizingNozzleAlphaAngle


class SizingNozzle(om.Group):
    """
    Nozzle sizing computations.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="connected_diffuser_id",
            default=None,
            desc="Identifier of the connected diffuser",
            allow_none=False,
        )
        self.options.declare(
            name="connected_air_inlet_id",
            default=None,
            desc="Identifier of the connected air inlet",
            allow_none=False,
        )
        self.options.declare(
            name="connected_heat_exchanger_id",
            default=None,
            desc="Identifier of the connected heat exchanger",
            allow_none=False,
        )
        self.options.declare(
            name="nozzle_id",
            default=None,
            desc="Identifier of the nozzle",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_diffuser_id = self.options["connected_diffuser_id"]
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]
        connected_air_inlet_id = self.options["connected_air_inlet_id"]
        nozzle_id = self.options["nozzle_id"]

        self.add_subsystem(
            "sizing_nozzle_length",
            SizingNozzleLength(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                connected_diffuser_id=connected_diffuser_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_outlet_inner_area",
            SizingNozzleExitArea(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                connected_air_inlet_id=connected_air_inlet_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_area_ratio",
            SizingNozzleAreaRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_exit_inner_dimension",
            SizingNozzleExitDimension(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_outer_dimension",
            SizingOuterDimension(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_inner_volume",
            SizingInnerVolume(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_outer_volume",
            SizingOuterVolume(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_weight",
            SizingNozzleWeight(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_entry_hydraulic_diameter",
            SizingEntryHydraulicDiameter(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_relative_roughness",
            SizingNozzleRelativeRoughness(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_exit_height_length_ratio",
            SizingNozzleExitHeightLengthRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_nozzle_alpha_angle",
            SizingNozzleAlphaAngle(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                connected_heat_exchanger_id=connected_heat_exchanger_id,
            ),
            promotes=["*"],
        )
