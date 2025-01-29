# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION, MULTI_TANK_FACTOR
import logging

_LOGGER = logging.getLogger(__name__)


class SizingGaseousHydrogenTankOuterDiameter(om.ExplicitComponent):
    """
    Outer diameter calculation fot the gaseous hydrogen tank.
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gaseous hydrogen tank",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="in_the_fuselage",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the gaseous hydrogen tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":diameter_height_ratio",
            val=0.9,
            desc="Fraction between the tank outer diameter and fuselage height",
        )

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:number_of_tank",
            val=1.0,
            desc="Number of gaseous hydrogen tank. "
            "Default set 1.0 for single tank in fuselage and outside fuselage uses.",
        )

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume",
            units="m**3",
            val=np.nan,
            desc="Capacity of the tank in terms of volume",
        )

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=1.06,
            desc="Outer diameter of the gaseous hydrogen tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]
        diameter_height_ratio = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":diameter_height_ratio"
        ]

        not_in_fuselage = (position == "wing_pod") or (position == "underbelly")

        nb_tank = inputs["data:propulsion:he_power_train:gaseous_hydrogen_tank:number_of_tank"]
        # multi_tank_factor divides the outer diameter with respect to the number of tanks.
        if not_in_fuselage:
            multi_tank_factor = 1.0
        else:
            multi_tank_factor = 1 / MULTI_TANK_FACTOR.get(int(nb_tank))

        d = (
            diameter_height_ratio
            * inputs["data:geometry:fuselage:maximum_height"]
            * multi_tank_factor
        )
        # This condition is to keep the tank as cylindrical as possible.
        positive_length = (
            inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":inner_volume"
            ]
            >= nb_tank * np.pi * d**3 / 6
        )

        if positive_length and not not_in_fuselage:
            outputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = d

        elif not positive_length and not not_in_fuselage:
            outputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = (
                np.cbrt(
                    6
                    * inputs[
                        "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                        + gaseous_hydrogen_tank_id
                        + ":inner_volume"
                    ]
                    / np.pi
                )
                * multi_tank_factor
            )

            _LOGGER.warning(msg="Possible Negative length!! Tank diameter adjust to proper size")

        elif not_in_fuselage:
            outputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = 0.5 * inputs["data:geometry:fuselage:maximum_height"]

            _LOGGER.warning(msg="Tank dimension fixed to reduce drag")

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]
        diameter_height_ratio = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":diameter_height_ratio"
        ]

        not_in_fuselage = (position == "wing_pod") or (position == "underbelly")

        nb_tank = inputs["data:propulsion:he_power_train:gaseous_hydrogen_tank:number_of_tank"]
        # multi_tank_factor divides the outer diameter with respect to the number of tanks.
        if not_in_fuselage:
            multi_tank_factor = 1.0
        else:
            multi_tank_factor = 1 / MULTI_TANK_FACTOR.get(int(nb_tank))

        d = (
            diameter_height_ratio
            * inputs["data:geometry:fuselage:maximum_height"]
            * multi_tank_factor
        )
        # This condition is to keep the tank as cylindrical as possible.
        positive_length = (
            inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":inner_volume"
            ]
            >= nb_tank * np.pi * d**3 / 6
        )

        if positive_length and not not_in_fuselage:
            partials[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:geometry:fuselage:maximum_height",
            ] = diameter_height_ratio * multi_tank_factor

            partials[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":diameter_height_ratio",
            ] = inputs["data:geometry:fuselage:maximum_height"] * multi_tank_factor

        elif not positive_length and not not_in_fuselage:
            partials[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":inner_volume",
            ] = (
                2
                * multi_tank_factor
                / np.pi
                * np.cbrt(
                    6
                    * inputs[
                        "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                        + gaseous_hydrogen_tank_id
                        + ":inner_volume"
                    ]
                    / np.pi
                )
                ** (-2)
            )

        elif not_in_fuselage:
            partials[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:geometry:fuselage:maximum_height",
            ] = 0.5
