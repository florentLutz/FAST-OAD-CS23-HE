# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION
import logging

_LOGGER = logging.getLogger(__name__)


class SizingCryogenicHydrogenTankThermalResistance(om.ExplicitComponent):
    """
    Computation of overall thermal resistance
    Reference material density are cite from: Hydrogen Storage for Aircraft Application Overview, NASA 2002
    """

    def initialize(self):

        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

    def setup(self):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        self.add_input(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":wall_thermal_resistance",
            units="K/W",
            val=np.nan,
            desc="Thermal resistance of the tank wall",
        )

        self.add_input(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_resistance",
            units="K/W",
            val=np.nan,
            desc="Thermal resistance of the the insulation layer",
        )

        self.add_output(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":thermal_resistance",
            units="K/W",
            val=50.0,
            desc="overall thermal resistance of the tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        outputs[input_prefix + ":thermal_resistance"] = (
            inputs[input_prefix + ":insulation:thermal_resistance"]
            + inputs[input_prefix + ":wall_thermal_resistance"]
        )
