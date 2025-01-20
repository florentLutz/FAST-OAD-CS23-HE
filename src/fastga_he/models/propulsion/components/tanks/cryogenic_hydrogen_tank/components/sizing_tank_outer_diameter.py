# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION
import logging

_LOGGER = logging.getLogger(__name__)
ADJUST_FACTOR = 1.0


class SizingCryogenicHydrogenTankOuterDiameter(om.ExplicitComponent):
    """
    Diameter check fot the hydrogen gas tank.
    """

    def initialize(self):

        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="in_the_fuselage",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the cryogenic hydrogen tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:diameter",
            units="m",
            val=np.nan,
            desc="Initial Outer diameter of the hydrogen gas tank input",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:length",
            val=0.0,
            units="m",
            desc="To avoid negative inner length",
        )

        self.add_output(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=1.06,
            desc="Outer diameter of the hydrogen gas tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]

        d = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:diameter"
        ]

        not_under_wing = position != "wing_pod"

        not_fit_in_fuselage = (
            inputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:diameter"
            ]
            > inputs["data:geometry:fuselage:maximum_height"]
        )

        positive_length = (
            inputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:length"
            ]
            >= 0
        )

        if not_under_wing and not_fit_in_fuselage and positive_length:

            outputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = (0.9 * inputs["data:geometry:fuselage:maximum_height"])

            _LOGGER.warning(
                msg="Tank dimension greater than fuselage!! Tank diameter adjust to proper size"
            )

        elif not positive_length:

            outputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = ADJUST_FACTOR * np.cbrt(
                d ** 3
                + 3
                * d ** 2
                * inputs[
                    "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                    + cryogenic_hydrogen_tank_id
                    + ":dimension:length"
                ]
                / 2
            )

            _LOGGER.warning(msg="Negative length!! Tank diameter adjust to proper size")

        elif position == "in_the_fuselage" and d >= (
            0.75 * inputs["data:geometry:fuselage:maximum_height"]
        ):
            outputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = (0.75 * inputs["data:geometry:fuselage:maximum_height"])

        else:

            outputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = d

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]

        not_under_wing = position != "wing_pod"

        d = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:diameter"
        ]

        not_fit_in_fuselage = (
            inputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:diameter"
            ]
            > inputs["data:geometry:fuselage:maximum_height"]
        )

        positive_length = (
            inputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:length"
            ]
            >= 0
        )

        if not_under_wing and not_fit_in_fuselage and positive_length:

            partials[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:geometry:fuselage:maximum_height",
            ] = 0.9

        elif not positive_length:
            partials[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:diameter",
            ] = (
                ADJUST_FACTOR
                * d
                * (
                    d
                    + inputs[
                        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                        + cryogenic_hydrogen_tank_id
                        + ":dimension:length"
                    ]
                )
                / (
                    d ** 3
                    + 3
                    * inputs[
                        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                        + cryogenic_hydrogen_tank_id
                        + ":dimension:length"
                    ]
                    * d ** 2
                    / 2
                )
                ** (2 / 3)
            )
            partials[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:length",
            ] = (
                ADJUST_FACTOR
                * 0.5
                * d ** 2
                / (
                    3
                    * d ** 2
                    * inputs[
                        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                        + cryogenic_hydrogen_tank_id
                        + ":dimension:length"
                    ]
                    / 2
                    + d ** 3
                )
                ** (2 / 3)
            )

        elif position == "in_the_fuselage" and d >= (
            0.75 * inputs["data:geometry:fuselage:maximum_height"]
        ):
            partials[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:geometry:fuselage:maximum_height",
            ] = 0.75

        else:

            partials[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:diameter",
            ] = 1.0
