# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION
import logging

_LOGGER = logging.getLogger(__name__)


class SizingCryogenicHydrogenTankInsulationThermalResistance(om.ExplicitComponent):
    """
    Computation of the thermal resistance of the insulation level
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
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the hydrogen gas tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_conductivity",
            units="W/m/K",
            val=np.nan,
            desc="Thermal conductivity of the insulation material",
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
            + ":insulation:thermal_resistance",
            units="K/W",
            val=50.0,
            desc="Thermal resistance of the the insulation layer",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        k = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_conductivity"
        ]

        d = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter"
        ]

        dw = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:wall_diameter"
        ]

        l = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:length"
        ]

        resistance_cylindrical = np.log(d / dw) / (2 * np.pi * l * k)

        resistance_spherical = (1 / d + 1 / dw) / (2 * np.pi * k)

        outputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_resistance"
        ] = (1 / resistance_cylindrical + 1 / resistance_spherical) ** -1

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        k = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_conductivity"
        ]

        d = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter"
        ]

        dw = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:wall_diameter"
        ]

        l = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:length"
        ]

        partials[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_resistance",
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_conductivity",
        ] = (-2 * np.pi * k**2 * (l / np.log(d / dw) + 1 / (1 / dw + 1 / d))) ** -1

        partials[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_resistance",
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:length",
        ] = -(
            (2 * np.pi * k * np.log(d / dw) * (l / np.log(d / dw) + 1 / (1 / dw + 1 / d)) ** 2)
            ** -1
        )

        partials[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_resistance",
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter",
        ] = (l / (d * np.log(d / dw) ** 2) - dw / (d + dw) + d * dw / (d + dw) ** 2) / (
            2 * np.pi * k * (l / np.log(d / dw) + d * dw / (d + dw)) ** 2
        )

        partials[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_resistance",
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:wall_diameter",
        ] = -(d / (d + dw) - d * dw / (d + dw) ** 2 + l / (dw * np.log(d / dw) ** 2)) / (
            2 * np.pi * k * (l / np.log(d / dw) + d * dw / (d + dw)) ** 2
        )
