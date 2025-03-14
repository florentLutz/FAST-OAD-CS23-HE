# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import logging

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION

_LOGGER = logging.getLogger(__name__)


class SizingH2FuelSystemPipeLength(om.ExplicitComponent):
    """
    The individual pipe length of the hydrogen fuel system, the number of lines is assumed to be
    equal to the number of output sources.
    """

    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="from_rear_to_center",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the hydrogen fuel system, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        self.options.declare(
            name="number_of_sources",
            default=1,
            types=int,
            desc="Number of connections at the output of the hydrogen fuel system, should always be "
            "power source",
            allow_none=False,
        )

    def setup(self):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        position = self.options["position"]
        number_of_sources = self.options["number_of_sources"]
        from_wing = (
            position == "from_front_to_wing"
            or position == "from_rear_to_wing"
            or position == "from_center_to_wing"
        )
        single_position = (
            position == "in_the_back" or position == "in_the_wing" or position == "at_center"
        )

        for i in range(number_of_sources):
            self.add_output(
                name="pipe_length_" + str(i + 1),
                units="m",
                val=1.0,
                desc="The pipe length for the output number " + str(i + 1),
            )

            if from_wing:
                self.add_input("data:geometry:wing:span", val=np.nan, units="m")
                self.add_input(
                    "data:propulsion:he_power_train:H2_fuel_system:"
                    + h2_fuel_system_id
                    + ":CG:y_ratio",
                    val=np.nan,
                    desc="Y position of the power source center of gravity as a ratio of the wing "
                    "half-span",
                )

            if not single_position:
                self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

                self.declare_partials("*", "*", method="exact")

            if single_position:
                self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

                self.declare_partials("*", "*", val=0.5)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        fuel_mass = inputs[
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":total_fuel_flowed"
        ]
        fuel_type = inputs[
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":fuel_type"
        ]

        if fuel_type == 1.0:
            self.rho_fuel = 718.9  # gasoline volume-mass [kg/m**3], cold worst case, Avgas
        elif fuel_type == 2.0:
            self.rho_fuel = 860.0  # Diesel volume-mass [kg/m**3], cold worst case
        elif fuel_type == 3.0:
            self.rho_fuel = 804.0  # Jet-A1 volume mass [kg/m**3], cold worst case
        else:
            self.rho_fuel = 718.9
            _LOGGER.warning("Fuel type %f does not exist, replaced by type 1!", fuel_type)

        outputs[
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":connected_volume"
        ] = fuel_mass / self.rho_fuel

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        partials[
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":connected_volume",
            "data:propulsion:he_power_train:fuel_system:"
            + h2_fuel_system_id
            + ":total_fuel_flowed",
        ] = 1.0 / self.rho_fuel
