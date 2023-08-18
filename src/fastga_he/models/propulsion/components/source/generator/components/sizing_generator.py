# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .cstr_generator import ConstraintsGenerator

from .sizing_diameter_scaling import SizingGeneratorDiameterScaling
from .sizing_diameter import SizingGeneratorDiameter
from .sizing_length_scaling import SizingGeneratorLengthScaling
from .sizing_length import SizingGeneratorLength
from .sizing_loss_coefficient_scaling import SizingGeneratorLossCoefficientScaling
from .sizing_loss_coefficient import SizingGeneratorLossCoefficient
from .sizing_resistance_scaling import SizingGeneratorPhaseResistanceScaling
from .sizing_resistance import SizingGeneratorPhaseResistance
from .sizing_torque_constant_scaling import SizingGeneratorTorqueConstantScaling
from .sizing_torque_constant import SizingGeneratorTorqueConstant
from .sizing_weight import SizingGeneratorWeight
from .sizing_generator_cg_x import SizingGeneratorCGX
from .sizing_generator_cg_y import SizingGeneratorCGY
from .sizing_generator_drag import SizingGeneratorDrag

from ..constants import POSSIBLE_POSITION


class SizingGenerator(om.Group):
    def initialize(self):
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the generator, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        generator_id = self.options["generator_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_generator",
            subsys=ConstraintsGenerator(generator_id=generator_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "diameter_scaling",
            SizingGeneratorDiameterScaling(generator_id=generator_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "diameter", SizingGeneratorDiameter(generator_id=generator_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "length_scaling",
            SizingGeneratorLengthScaling(generator_id=generator_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "length", SizingGeneratorLength(generator_id=generator_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "weight", SizingGeneratorWeight(generator_id=generator_id), promotes=["data:*"]
        )

        self.add_subsystem(
            "resistance_scaling",
            SizingGeneratorPhaseResistanceScaling(generator_id=generator_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "resistance",
            SizingGeneratorPhaseResistance(generator_id=generator_id),
            promotes=["data:*"],
        )

        self.add_subsystem(
            "torque_constant_scaling",
            SizingGeneratorTorqueConstantScaling(generator_id=generator_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "torque_constant",
            SizingGeneratorTorqueConstant(generator_id=generator_id),
            promotes=["data:*"],
        )

        self.add_subsystem(
            "loss_coefficients_scaling",
            SizingGeneratorLossCoefficientScaling(generator_id=generator_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "loss_coefficients",
            SizingGeneratorLossCoefficient(generator_id=generator_id),
            promotes=["data:*"],
        )

        self.add_subsystem(
            "generator_cg_x",
            SizingGeneratorCGX(generator_id=generator_id, position=position),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "generator_cg_y",
            SizingGeneratorCGY(generator_id=generator_id, position=position),
            promotes=["data:*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "generator_drag_ls" if low_speed_aero else "generator_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingGeneratorDrag(
                    generator_id=generator_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
