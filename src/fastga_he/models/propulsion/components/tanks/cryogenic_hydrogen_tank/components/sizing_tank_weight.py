# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION
import logging

_LOGGER = logging.getLogger(__name__)


class SizingCryogenicHydrogenTankWeight(om.ExplicitComponent):
    """
    Computation of the weight of the tank. The very simplistic approach we will use is to say
    that weight of tank is the weight of the tank itself.
    Reference material density are cite from: Hydrogen Storage for Aircraft Application Overview, NASA 2002
    """

    def initialize(self):
        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

        self.options.declare(
            "structure_factor",
            default=1.0,
            desc="Structure factor to consider other part of the tank system",
        )

    def setup(self):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":inner_volume",
            units="m**3",
            val=np.nan,
            desc="Capacity of the tank in terms of volume",
        )

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
            + ":material:density",
            units="kg/m**3",
            val=7860.0,
            desc="Choice of the tank material,Some reference: Steel(ASTM-A514):7860, "
            "Aluminum(2014-T6):2800, Titanium(6%Al,4%V):4460, Carbon Composite:1530",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:material_density",
            units="kg/m**3",
            val=120.0,
            desc="Choice of the insulation material,Some reference: "
            "Evacuated aluminum foil & glass paper laminate:120,"
            "Evacuated silica powder:160",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:wall_diameter",
            units="m",
            val=np.nan,
            desc="Wall diameter of the hydrogen tank without insulation layer",
        )

        self.add_output(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":mass",
            units="kg",
            val=20.0,
            desc="Weight of the hydrogen gas tanks",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        structure_factor = self.options["structure_factor"]

        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        wall_density = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":material:density"
        ]

        insulation_density = inputs[input_prefix + ":insulation:material_density"]

        d = inputs[input_prefix + ":dimension:outer_diameter"]

        dw = inputs[input_prefix + ":dimension:wall_diameter"]

        l = inputs[input_prefix + ":dimension:length"]

        outputs[input_prefix + ":mass"] = structure_factor * (
            wall_density
            * (np.pi * dw**3 / 6 + np.pi * dw**2 * l / 4 - inputs[input_prefix + ":inner_volume"])
            + insulation_density * (np.pi * (d**3 - dw**3) / 6 + np.pi * (d**2 - dw**2) * l / 4)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        structure_factor = self.options["structure_factor"]

        d = inputs[input_prefix + ":dimension:outer_diameter"]

        dw = inputs[input_prefix + ":dimension:wall_diameter"]

        l = inputs[input_prefix + ":dimension:length"]

        wall_density = inputs[input_prefix + ":material:density"]

        insulation_density = inputs[input_prefix + ":insulation:material_density"]

        partials[
            input_prefix + ":mass",
            input_prefix + ":insulation:material_density",
        ] = structure_factor * (np.pi * (d**3 - dw**3) / 6 + np.pi * (d**2 - dw**2) * l / 4)

        partials[
            input_prefix + ":mass",
            input_prefix + ":material:density",
        ] = structure_factor * (
            np.pi * dw**3 / 6 + np.pi * dw**2 / 4 * l - inputs[input_prefix + ":inner_volume"]
        )

        partials[
            input_prefix + ":mass",
            input_prefix + ":dimension:length",
        ] = structure_factor * (
            wall_density * np.pi * dw**2 / 4 + insulation_density * np.pi * (d**2 - dw**2) / 4
        )

        partials[
            input_prefix + ":mass",
            input_prefix + ":inner_volume",
        ] = -wall_density * structure_factor

        partials[
            input_prefix + ":mass",
            input_prefix + ":dimension:outer_diameter",
        ] = insulation_density * np.pi / 2 * (d**2 + d * l) * structure_factor

        partials[
            input_prefix + ":mass",
            input_prefix + ":dimension:wall_diameter",
        ] = np.pi / 2 * (wall_density - insulation_density) * (dw**2 + dw * l) * structure_factor
