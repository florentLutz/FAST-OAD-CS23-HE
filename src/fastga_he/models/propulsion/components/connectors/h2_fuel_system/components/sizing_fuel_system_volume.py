# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging

import openmdao.api as om
import numpy as np

_LOGGER = logging.getLogger(__name__)


class SizingFuelSystemCapacityVolume(om.ExplicitComponent):
    """
    Not really the capacity in terms of volume of the hydrogen fuel system but rather the sum of the
    volume of all connected tanks; The reason being that the mass we tak in input is the sum of
    the mass going in all connected tank at all point of the flight.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rho_fuel = None

    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )

    def setup(self):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        self.add_input(
            "data:propulsion:he_power_train:fuel_system:"
            + h2_fuel_system_id
            + ":total_fuel_flowed",
            units="kg",
            val=np.nan,
            desc="Total amount of fuel that flowed through the system",
        )
        self.add_input(
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":fuel_type",
            val=1.0,
            desc="Type of fuel flowing in the system, 1.0 - gasoline, 2.0 - Diesel, 3.0 - Jet A1",
        )

        self.add_output(
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":connected_volume",
            units="m**3",
            val=0.07,
            desc="Capacity of the connected tank in terms of volume",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:fuel_system:"
            + h2_fuel_system_id
            + ":connected_volume",
            wrt="data:propulsion:he_power_train:fuel_system:"
            + h2_fuel_system_id
            + ":total_fuel_flowed",
            method="exact",
        )

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
