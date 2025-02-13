# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

HYDROGEN_GAS_CONSTANT = 4157.2  # (N.m/K.kg)
STORAGE_TEMPERATURE = 300.0  # (K)


class SizingGaseousHydrogenTankInnerVolume(om.ExplicitComponent):
    """
    Computation of the volume of hydrogen to be stored in the tank in specific temperature and
    pressure condition, performed under ideal gas assumption.
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity",
            units="kg",
            val=np.nan,
            desc="Capacity of the tank in terms of weight",
        )

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":tank_pressure",
            val=np.nan,
            units="Pa",
            desc="gaseous hydrogen tank static pressure, "
            "convention storage pressure in industry: 35 MPa, 70 MPa ",
        )

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
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
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        fuel_mass = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity"
        ]

        tank_pressure = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":tank_pressure"
        ]

        z = 0.99704 + 6.4149e-9 * tank_pressure
        # Hydrogen gas compressibility factor :cite:`bolz:1973`.

        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume"
        ] = z * HYDROGEN_GAS_CONSTANT * fuel_mass * STORAGE_TEMPERATURE / tank_pressure

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        fuel_mass = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity"
        ]

        tank_pressure = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":tank_pressure"
        ]

        z = 0.99704 + 6.4149e-9 * tank_pressure
        # Hydrogen gas compressibility factor :cite:`bolz:1973`.

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity",
        ] = z * HYDROGEN_GAS_CONSTANT * STORAGE_TEMPERATURE / tank_pressure

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":tank_pressure",
        ] = -0.99704 * HYDROGEN_GAS_CONSTANT * fuel_mass * STORAGE_TEMPERATURE / tank_pressure**2
