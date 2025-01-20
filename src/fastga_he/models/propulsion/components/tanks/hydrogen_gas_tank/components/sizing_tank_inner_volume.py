# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging

import openmdao.api as om
import numpy as np

_LOGGER = logging.getLogger(__name__)
HYDROGEN_GAS_CONSTANT = 4157.2  # (N.m/K.kg)


class SizingHydrogenGasTankInnerVolume(om.ExplicitComponent):
    """
    Computation of the volume of fuel to be stored in the tank in standard temperature condition (300K).
    Calculate under ideal gas assumption.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rho_fuel = None

    def initialize(self):
        self.options.declare(
            name="hydrogen_gas_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":capacity",
            units="kg",
            val=np.nan,
            desc="Capacity of the tank in terms of weight",
        )

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":tank_pressure",
            val=np.nan,
            units="Pa",
            desc="Hydrogen gas tank static pressure",
        )

        self.add_input(
            "tank_temperature",
            val=300.0,
            units="K",
            desc="Hydrogen gas tank temperature",
        )

        self.add_output(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":inner_volume",
            units="m**3",
            val=15.0,
            desc="Capacity of the tank in terms of volume",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        fuel_mass = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:" + hydrogen_gas_tank_id + ":capacity"
        ]

        tank_pressure = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":tank_pressure"
        ]

        z = 0.99704 + 6.4149e-9 * tank_pressure  # compressibility correction

        tank_temperature = inputs["tank_temperature"]

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":inner_volume"
        ] = z * HYDROGEN_GAS_CONSTANT * fuel_mass * tank_temperature / tank_pressure

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        fuel_mass = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:" + hydrogen_gas_tank_id + ":capacity"
        ]

        tank_pressure = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":tank_pressure"
        ]

        z = 0.99704 + 6.4149e-9 * tank_pressure  # compressibility correction

        tank_temperature = inputs["tank_temperature"]

        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":inner_volume",
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":capacity",
        ] = z * HYDROGEN_GAS_CONSTANT * tank_temperature / tank_pressure

        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":inner_volume",
            "tank_temperature",
        ] = z * HYDROGEN_GAS_CONSTANT * fuel_mass / tank_pressure

        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":inner_volume",
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":tank_pressure",
        ] = -0.99704 * HYDROGEN_GAS_CONSTANT * fuel_mass * tank_temperature / tank_pressure**2
