# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging

import openmdao.api as om
import numpy as np

_LOGGER = logging.getLogger(__name__)


class SizingFuelTankVolume(om.ExplicitComponent):
    """
    Computation of the volume of fuel to be stored in the tank. Assumes that there is no
    significant difference between tank volume and fuel volume, can be change easily later. Also,
    the fuel type will be an input of this component though I am not pleased with this idea since
    it should depend on what type of fuel burning engine it is connected to, thus the info should
    come from there.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rho_fuel = None

    def initialize(self):

        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):

        fuel_tank_id = self.options["fuel_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity",
            units="kg",
            val=np.nan,
            desc="Capacity of the tank in terms of weight",
        )
        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_type",
            val=1.0,
            desc="Type of fuel stored in the tank, 1.0 - gasoline, 2.0 - Diesel, 3.0 - Jet A1",
        )

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
            units="m**3",
            val=0.07,
            desc="Capacity of the tank in terms of volume",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
            wrt="data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]

        fuel_mass = inputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity"]
        fuel_type = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_type"
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

        outputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume"] = (
            fuel_mass / self.rho_fuel
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]

        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity",
        ] = (
            1.0 / self.rho_fuel
        )
