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
    The outer diameter of the gaseous hydrogen tank is based on
    the maximum fuselage height and the number of tanks.
    For multi-tank stack scenarios,
    all the tanks are assumed with the same diameter and using the table provided by:
    :cite:`kravitz:1967`
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
            default="in_the_cabin",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the gaseous hydrogen tank, "
            "possible position include " + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]
        in_fuselage = position == "in_the_cabin" or position == "in_the_back"

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":diameter_height_ratio",
            val=0.9,
            desc="Fraction between the tank outer diameter and fuselage height",
        )
        if in_fuselage:
            self.add_input(
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":number_of_tank",
                val=1.0,
                desc="Number of gaseous hydrogen tank in a stack in fuselage. "
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
        # Ratio between the tank outer diameter and fuselage height
        not_in_fuselage = (position == "wing_pod") or (position == "underbelly")

        inner_volume = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume"
        ]

        if not_in_fuselage:
            multi_tank_factor = 1.0
            nb_tank = 1.0
            _LOGGER.info(
                msg="Number of tank per stack fixed to 1 for all outside fuselage position"
            )
        else:
            nb_tank = inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":number_of_tank"
            ]
            multi_tank_factor = 1 / MULTI_TANK_FACTOR.get(int(nb_tank))
        # multi_tank_factor divides the outer diameter with respect to the number of tanks.
        d_tank = (
            diameter_height_ratio
            * inputs["data:geometry:fuselage:maximum_height"]
            * multi_tank_factor
        )
        # This condition is to keep the tank as cylindrical as possible.
        has_sufficient_volume = inner_volume >= nb_tank * np.pi * d_tank**3 / 6

        if has_sufficient_volume and not not_in_fuselage:
            outputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = d_tank

        elif not has_sufficient_volume and not not_in_fuselage:
            outputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = np.cbrt(6 * inner_volume / np.pi) * multi_tank_factor

            _LOGGER.warning(
                msg="Inconsistent tank length for tank(s) "
                + gaseous_hydrogen_tank_id
                + " due to the diameter being too large for the required tank capacity, "
                "suggest to reduce the diameter."
            )

        elif not_in_fuselage:
            outputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ] = 0.2 * inputs["data:geometry:fuselage:maximum_height"]
            # Ratio inspired by the size of fighter jet external fuel tank
            _LOGGER.info(msg="Tank diameter fixed as 20% of fuselage maximum height")

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]
        diameter_height_ratio = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":diameter_height_ratio"
        ]
        # Ratio between the tank outer diameter and fuselage height
        inner_volume = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume"
        ]

        not_in_fuselage = (position == "wing_pod") or (position == "underbelly")

        # multi_tank_factor divides the outer diameter with respect to the number of tanks.
        if not_in_fuselage:
            multi_tank_factor = 1.0
            nb_tank = 1.0
        else:
            nb_tank = inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":number_of_tank"
            ]
            multi_tank_factor = 1 / MULTI_TANK_FACTOR.get(int(nb_tank))

        d_tank = (
            diameter_height_ratio
            * inputs["data:geometry:fuselage:maximum_height"]
            * multi_tank_factor
        )
        # This condition is to keep the tank as cylindrical as possible.
        has_sufficient_volume = inner_volume >= nb_tank * np.pi * d_tank**3 / 6

        if has_sufficient_volume and not not_in_fuselage:
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

        elif not has_sufficient_volume and not not_in_fuselage:
            partials[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":inner_volume",
            ] = 2 * multi_tank_factor / np.pi * np.cbrt(6 * inner_volume / np.pi) ** (-2)

        elif not_in_fuselage:
            partials[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter",
                "data:geometry:fuselage:maximum_height",
            ] = 0.2
