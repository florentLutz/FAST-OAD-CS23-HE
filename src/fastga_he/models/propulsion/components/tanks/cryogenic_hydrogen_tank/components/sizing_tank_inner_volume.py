# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


import openmdao.api as om
import numpy as np


LIQUID_HYDROGEN_DENSITY = 70.85  # kg/m**3


class SizingCryogenicHydrogenTankInnerVolume(om.ExplicitComponent):
    """
    Computation of the volume of fuel to be stored in the tank in standard temperature condition (300K).
    Calculate under ideal gas assumption.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rho_fuel = None

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
            + ":capacity",
            units="kg",
            val=np.nan,
            desc="Capacity of the tank in terms of weight",
        )

        self.add_output(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":inner_volume",
            units="m**3",
            val=15.0,
            desc="Capacity of the tank in terms of volume",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            val=1 / LIQUID_HYDROGEN_DENSITY,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        fuel_mass = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":capacity"
        ]

        outputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":inner_volume"
        ] = fuel_mass / LIQUID_HYDROGEN_DENSITY
