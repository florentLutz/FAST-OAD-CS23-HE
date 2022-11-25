# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

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

from ..constants import SUBMODEL_CONSTRAINTS_PMSM


class SizingPMSM(om.Group):
    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        option_motor_id = {"motor_id": motor_id}

        self.add_subsystem(
            name="constraints_dc_dc_converter",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_PMSM, options=option_motor_id
            ),
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
