# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .sizing_diameter import SizingStatorDiameter
from .sizing_active_length import SizingActiveLength
from .sizing_rotor_diameter import SizingRotorDiameter
from .sizing_stator_yoke import SizingStatorYokeHeight
from .sizing_slot_width import SizingSlotWidth
from .sizing_slot_height import SizingSlotHeight
from .sizing_slot_section import SizingSlotSection
from .sizing_conductor_section import SizingConductorSection
from .sizing_conductor_length import SizingConductorLength
from .sizing_conductors_number import SizingConductorsNumber
from .sizing_winding_resistivity import SizingWindingResistivity
from .sizing_resistance import SizingResistance
from .sizing_external_stator_diameter import SizingExtStatorDiameter
from .sizing_stator_core_weight import SizingStatorCoreWeight
from .sizing_winding_stator_weight import SizingStatorWindingWeight
from .sizing_rotor_weight import SizingRotorWeight
from .sizing_frame_weight import SizingFrameWeight
from .sizing_pmsm_weight import SizingMotorWeight
from .sizing_pmsm_cg_x import SizingPMSMCGX
from .sizing_pmsm_cg_y import SizingPMSMCGY
from .sizing_pmsm_drag import SizingPMSMDrag
from .sizing_ratio_x2p import SizingRatioX2p
from .sizing_tooth_ratio import SizingToothRatio

from .cstr_ac_pmsm import ConstraintsPMSM

from ..constants import POSSIBLE_POSITION


class SizingACPMSM(om.Group):
    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the pmsm, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_pmsm",
            subsys=ConstraintsPMSM(pmsm_id=pmsm_id),
            promotes=["*"],
        )

        self.add_subsystem("diameter", SizingStatorDiameter(pmsm_id=pmsm_id), promotes=["data:*"])

        self.add_subsystem("length", SizingActiveLength(pmsm_id=pmsm_id), promotes=["data:*"])

        self.add_subsystem(
            "rotor_diameter", SizingRotorDiameter(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem("ratio_x2p", SizingRatioX2p(pmsm_id=pmsm_id), promotes=["data:*"])

        self.add_subsystem("tooth_ratio", SizingToothRatio(pmsm_id=pmsm_id), promotes=["data:*"])

        self.add_subsystem(
            "yoke_height", SizingStatorYokeHeight(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "conductor_number", SizingConductorsNumber(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem("slot_width", SizingSlotWidth(pmsm_id=pmsm_id), promotes=["data:*"])

        self.add_subsystem("slot_height", SizingSlotHeight(pmsm_id=pmsm_id), promotes=["data:*"])

        self.add_subsystem(
            "slot_cross_section", SizingSlotSection(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "conductor_cross_section", SizingConductorSection(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "conductor_length", SizingConductorLength(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "winding_resistivity", SizingWindingResistivity(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "electric_resistance", SizingResistance(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "stator_external_diameter",
            SizingExtStatorDiameter(pmsm_id=pmsm_id),
            promotes=["data:*"],
        )

        self.add_subsystem(
            "stator_core_weight", SizingStatorCoreWeight(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "winding_weight", SizingStatorWindingWeight(pmsm_id=pmsm_id), promotes=["data:*"]
        )

        self.add_subsystem("rotor_weight", SizingRotorWeight(pmsm_id=pmsm_id), promotes=["data:*"])

        self.add_subsystem("frame_weight", SizingFrameWeight(pmsm_id=pmsm_id), promotes=["data:*"])

        self.add_subsystem("mass", SizingMotorWeight(pmsm_id=pmsm_id), promotes=["data:*"])

        self.add_subsystem(
            "pmsm_cg_x", SizingPMSMCGX(pmsm_id=pmsm_id, position=position), promotes=["data:*"]
        )
        self.add_subsystem(
            "pmsm_cg_y", SizingPMSMCGY(pmsm_id=pmsm_id, position=position), promotes=["data:*"]
        )
        for low_speed_aero in [True, False]:
            system_name = "pmsm_drag_ls" if low_speed_aero else "pmsm_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingPMSMDrag(
                    pmsm_id=pmsm_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
