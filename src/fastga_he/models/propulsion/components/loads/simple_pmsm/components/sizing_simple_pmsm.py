# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_diameter_scaling import SizingMotorDiameterScaling
from .sizing_diameter import SizingMotorDiameter
from .sizing_length_scaling import SizingMotorLengthScaling
from .sizing_length import SizingMotorLength
from .sizing_weight import SizingMotorWeight
from .sizing_resistance_scaling import SizingMotorPhaseResistanceScaling
from .sizing_resistance import SizingMotorPhaseResistance
from .sizing_torque_constant_scaling import SizingMotorTorqueConstantScaling
from .sizing_torque_constant import SizingMotorTorqueConstant
from .sizing_loss_coefficient_scaling import SizingMotorLossCoefficientScaling
from .sizing_loss_coefficient import SizingMotorLossCoefficient
from .sizing_simple_pmsm_cg_x import SizingSimplePMSMCGX
from .sizing_simple_pmsm_cg_y import SizingSimplePMSMCGY
from .sizing_simple_pmsm_drag import SizingSimplePMSMDrag
from .sizing_power_density import SizingPowerDensity
from .cstr_simple_pmsm import ConstraintsSimplePMSM

from ..constants import POSSIBLE_POSITION


class SizingSimplePMSM(om.Group):
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
            subsys=ConstraintsSimplePMSM(motor_id=motor_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "diameter_scaling", SizingMotorDiameterScaling(motor_id=motor_id), promotes=["data:*"]
        )
        self.add_subsystem("diameter", SizingMotorDiameter(motor_id=motor_id), promotes=["data:*"])

        self.add_subsystem(
            "length_scaling", SizingMotorLengthScaling(motor_id=motor_id), promotes=["data:*"]
        )
        self.add_subsystem("length", SizingMotorLength(motor_id=motor_id), promotes=["data:*"])

        self.add_subsystem("weight", SizingMotorWeight(motor_id=motor_id), promotes=["data:*"])

        self.add_subsystem(
            "resistance_scaling",
            SizingMotorPhaseResistanceScaling(motor_id=motor_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "resistance", SizingMotorPhaseResistance(motor_id=motor_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "torque_constant_scaling",
            SizingMotorTorqueConstantScaling(motor_id=motor_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "torque_constant", SizingMotorTorqueConstant(motor_id=motor_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "loss_coefficients_scaling",
            SizingMotorLossCoefficientScaling(motor_id=motor_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "loss_coefficients", SizingMotorLossCoefficient(motor_id=motor_id), promotes=["data:*"]
        )
        self.add_subsystem(
            "pmsm_cg_x",
            SizingSimplePMSMCGX(motor_id=motor_id, position=position),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "pmsm_cg_y",
            SizingSimplePMSMCGY(motor_id=motor_id, position=position),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "power_density", SizingPowerDensity(motor_id=motor_id), promotes=["data:*"]
        )
        for low_speed_aero in [True, False]:
            system_name = "pmsm_drag_ls" if low_speed_aero else "pmsm_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingSimplePMSMDrag(
                    motor_id=motor_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
