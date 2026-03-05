# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .sizing_fin_geometry import SizingHeatExchangerFinGeometry
from .sizing_total_transfer_area_volume_ratio import SizingTotalTransferAreaVolumeRatio
from .sizing_fin_geometry_factor import SizingHeatExchangerFinFactor
from .sizing_heat_exchanger_separating_plate_layer_count import (
    SizingHeatExchangerSeparatingPlateLayerCount,
)
from .sizing_heat_exchanger_no_flow_length import SizingHeatExchangerNoFlowLength
from .sizing_fin_hydraulic_diameter import SizingHeatExchangerFinHydraulicDiameter


class SizingHeatExchanger(om.Group):
    """
    Heat exchanger sizing computations.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_subsystem(
            "sizing_fin_geometry",
            SizingHeatExchangerFinGeometry(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_total_transfer_area_volume_ratio",
            SizingTotalTransferAreaVolumeRatio(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_fin_geometry_factor",
            SizingHeatExchangerFinFactor(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_heat_exchanger_separating_plate_layer_count",
            SizingHeatExchangerSeparatingPlateLayerCount(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_heat_exchanger_no_flow_length",
            SizingHeatExchangerNoFlowLength(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_fin_hydraulic_diameter",
            SizingHeatExchangerFinHydraulicDiameter(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
