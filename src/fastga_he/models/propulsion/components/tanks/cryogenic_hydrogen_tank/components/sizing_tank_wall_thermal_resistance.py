# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION
import logging

_LOGGER = logging.getLogger(__name__)


class SizingCryogenicHydrogenTankWallThermalResistance(om.ExplicitComponent):
    """
    Computation of the overall thermal resistance of the wall itself
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
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:length",
            val=np.nan,
            units="m",
            desc="Value of the length of the tank in the x-direction",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:inner_diameter",
            units="m",
            val=np.nan,
            desc="Inner diameter of the hydrogen tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":material:thermal_conductivity",
            units="W/m/K",
            val=np.nan,
            desc="Thermal conductivity of the tank wall material",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:wall_diameter",
            units="m",
            val=np.nan,
            desc="Wall diameter of the hydrogen tank",
        )

        self.add_output(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":wall_thermal_resistance",
            units="K/W",
            val=50.0,
            desc="Thermal resistance of the tank wall ",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        k = inputs[input_prefix + ":material:thermal_conductivity"]

        dw = inputs[input_prefix + ":dimension:wall_diameter"]

        din = inputs[input_prefix + ":dimension:inner_diameter"]

        l = inputs[input_prefix + ":dimension:length"]

        resistance_cylindrical = np.log(dw / din) / (2 * np.pi * l * k)

        resistance_spherical = (1 / dw + 1 / din) / (2 * np.pi * k)

        outputs[input_prefix + ":wall_thermal_resistance"] = (
            1 / resistance_cylindrical + 1 / resistance_spherical
        ) ** -1

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        k = inputs[input_prefix + ":material:thermal_conductivity"]

        dw = inputs[input_prefix + ":dimension:wall_diameter"]

        din = inputs[input_prefix + ":dimension:inner_diameter"]

        l = inputs[input_prefix + ":dimension:length"]

        partials[
            input_prefix + ":wall_thermal_resistance",
            input_prefix + ":material:thermal_conductivity",
        ] = (-2 * np.pi * k**2 * (l / np.log(dw / din) + 1 / (1 / dw + 1 / din))) ** -1

        partials[input_prefix + ":wall_thermal_resistance", input_prefix + ":dimension:length"] = -(
            (2 * np.pi * k) ** -1
        ) / (np.log(dw / din) * (l / np.log(dw / din) + 1 / (1 / dw + 1 / din)) ** 2)

        partials[
            input_prefix + ":wall_thermal_resistance",
            input_prefix + ":dimension:wall_diameter",
        ] = (l / (dw * np.log(dw / din) ** 2) - din / (dw + din) + dw * din / (dw + din) ** 2) / (
            2 * np.pi * k * (l / np.log(dw / din) + dw * din / (dw + din)) ** 2
        )

        partials[
            input_prefix + ":wall_thermal_resistance",
            input_prefix + ":dimension:inner_diameter",
        ] = -(dw / (dw + din) - dw * din / (dw + din) ** 2 + l / (din * np.log(dw / din) ** 2)) / (
            2 * np.pi * k * (l / np.log(dw / din) + dw * din / (dw + din)) ** 2
        )
