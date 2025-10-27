# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .sizing_bore_diameter import SizingStatorBoreDiameter
from .sizing_active_length import SizingActiveLength
from .sizing_frame_dimension import SizingFrameDimension
from .sizing_rotor_diameter import SizingRotorDiameter
from .sizing_shaft_diameter import SizingShaftDiameter
from .sizing_stator_yoke import SizingStatorYokeHeight
from .sizing_slot_width import SizingSlotWidth
from .sizing_slot_height import SizingSlotHeight
from .sizing_slot_section_area import SizingSlotSectionArea
from .sizing_conductor_section_area_per_slot import SizingConductorSectionAreaPerSlot
from .sizing_single_conductor_cable_length import SizingSingleConductorCableLength
from .sizing_conductor_slot_number import SizingConductorSlotNumber
from .sizing_reference_resistance import SizingReferenceResistance
from .sizing_external_stator_diameter import SizingExternalStatorDiameter
from .sizing_stator_core_weight import SizingStatorCoreWeight
from .sizing_winding_stator_weight import SizingStatorWindingWeight
from .sizing_rotor_weight import SizingRotorWeight
from .sizing_frame_weight import SizingFrameWeight
from .sizing_sm_pmsm_weight import SizingMotorWeight
from .sizing_sm_pmsm_cg_x import SizingSMPMSMCGX
from .sizing_sm_pmsm_cg_y import SizingSMPMSMCGY
from .sizing_sm_pmsm_drag import SizingSMPMSMDrag
from .sizing_ratio_x2p import SizingRatioX2p
from .sizing_tooth_ratio import SizingToothRatio

from .cstr_sm_pmsm import ConstraintsSMPMSM

from ..constants import POSSIBLE_POSITION


class SizingSMPMSM(om.Group):
    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
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
        motor_id = self.options["motor_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_pmsm",
            subsys=ConstraintsSMPMSM(motor_id=motor_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "bore_diameter", SizingStatorBoreDiameter(motor_id=motor_id), promotes=["*"]
        )

        self.add_subsystem("active_length", SizingActiveLength(motor_id=motor_id), promotes=["*"])

        self.add_subsystem("rotor_diameter", SizingRotorDiameter(motor_id=motor_id), promotes=["*"])

        self.add_subsystem("shaft_diameter", SizingShaftDiameter(motor_id=motor_id), promotes=["*"])

        self.add_subsystem("ratio_x2p", SizingRatioX2p(motor_id=motor_id), promotes=["*"])

        self.add_subsystem("tooth_ratio", SizingToothRatio(motor_id=motor_id), promotes=["*"])

        self.add_subsystem("yoke_height", SizingStatorYokeHeight(motor_id=motor_id), promotes=["*"])

        self.add_subsystem(
            "conductor_number", SizingConductorSlotNumber(motor_id=motor_id), promotes=["*"]
        )

        self.add_subsystem("slot_width", SizingSlotWidth(motor_id=motor_id), promotes=["*"])

        self.add_subsystem("slot_height", SizingSlotHeight(motor_id=motor_id), promotes=["*"])

        self.add_subsystem(
            "slot_cross_section", SizingSlotSectionArea(motor_id=motor_id), promotes=["*"]
        )

        self.add_subsystem(
            "conductor_cross_section",
            SizingConductorSectionAreaPerSlot(motor_id=motor_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "single_cable_length",
            SizingSingleConductorCableLength(motor_id=motor_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "reference_resistance",
            SizingReferenceResistance(motor_id=motor_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "stator_external_diameter",
            SizingExternalStatorDiameter(motor_id=motor_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "stator_core_weight", SizingStatorCoreWeight(motor_id=motor_id), promotes=["*"]
        )

        self.add_subsystem(
            "wire_winding_weight", SizingStatorWindingWeight(motor_id=motor_id), promotes=["*"]
        )

        self.add_subsystem("rotor_weight", SizingRotorWeight(motor_id=motor_id), promotes=["*"])

        self.add_subsystem(
            "frame_dimension", SizingFrameDimension(motor_id=motor_id), promotes=["*"]
        )

        self.add_subsystem("frame_weight", SizingFrameWeight(motor_id=motor_id), promotes=["*"])

        self.add_subsystem("weight", SizingMotorWeight(motor_id=motor_id), promotes=["*"])

        self.add_subsystem(
            "pmsm_cg_x", SizingSMPMSMCGX(motor_id=motor_id, position=position), promotes=["*"]
        )
        self.add_subsystem(
            "pmsm_cg_y", SizingSMPMSMCGY(motor_id=motor_id, position=position), promotes=["*"]
        )
        for low_speed_aero in [True, False]:
            system_name = "pmsm_drag_ls" if low_speed_aero else "pmsm_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingSMPMSMDrag(
                    motor_id=motor_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
