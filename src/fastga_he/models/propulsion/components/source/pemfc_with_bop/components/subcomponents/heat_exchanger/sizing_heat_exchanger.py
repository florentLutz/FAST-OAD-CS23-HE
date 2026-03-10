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
from .sizing_heat_exchanger_flow_length import SizingHeatExchangerFlowLength
from .sizing_free_flow_frontal_area_ratio import SizingFreeFlowFrontalAreaRatio
from .sizing_heat_exchanger_plate_weight import SizingHeatExchangerPlateWeight
from .sizing_heat_exchanger_channel_weight import SizingHeatExchangerChannelWeight
from .sizing_heat_exchanger_coolant_volume import SizingHeatExchangerCoolantVolume
from .sizing_heat_exchanger_dry_weight import SizingHeatExchangerDryWeight


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
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_subsystem(
            "sizing_fin_geometry",
            SizingHeatExchangerFinGeometry(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_total_transfer_area_volume_ratio",
            SizingTotalTransferAreaVolumeRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_fin_geometry_factor",
            SizingHeatExchangerFinFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_heat_exchanger_separating_plate_layer_count",
            SizingHeatExchangerSeparatingPlateLayerCount(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_heat_exchanger_no_flow_length",
            SizingHeatExchangerNoFlowLength(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_fin_hydraulic_diameter",
            SizingHeatExchangerFinHydraulicDiameter(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_heat_exchanger_flow_length",
            SizingHeatExchangerFlowLength(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_free_flow_frontal_area_ratio",
            SizingFreeFlowFrontalAreaRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_heat_exchanger_plate_weight",
            SizingHeatExchangerPlateWeight(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_heat_exchanger_channel_weight",
            SizingHeatExchangerChannelWeight(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_heat_exchanger_fluid_weight",
            SizingHeatExchangerCoolantVolume(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_heat_exchanger_dry_weight",
            SizingHeatExchangerDryWeight(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
