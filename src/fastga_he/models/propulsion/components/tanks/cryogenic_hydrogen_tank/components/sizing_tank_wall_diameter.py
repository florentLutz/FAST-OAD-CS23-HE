# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION
import logging

_LOGGER = logging.getLogger(__name__)
ADJUST_FACTOR = 1.0


class SizingCryogenicHydrogenTankWallDiameter(om.ExplicitComponent):
    """
    Wall diameter of the hydrogen gas tank, not include the insulation layer.
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
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the hydrogen tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:insulation_thickness",
            units="m",
            val=np.nan,
            desc="Insulation layer thickness of the cryogenic hydrogen tank",
        )

        self.add_output(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:wall_diameter",
            units="m",
            val=1.0,
            desc="Wall diameter of the hydrogen tank without insulation layer",
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter",
            method="exact",
            val=1.0,
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:insulation_thickness",
            method="exact",
            val=-2.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        outputs[input_prefix + ":dimension:wall_diameter"] = (
            inputs[input_prefix + ":dimension:outer_diameter"]
            - 2 * inputs[input_prefix + ":dimension:insulation_thickness"]
        )
