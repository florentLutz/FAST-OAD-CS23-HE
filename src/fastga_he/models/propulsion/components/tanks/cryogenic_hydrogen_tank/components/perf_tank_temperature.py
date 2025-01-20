# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

LIQUID_HYDROGEN_LATENT_HEAT_OF_VAPORIZATION = 904.0  # J.mol
GAS_CONSTANT = 8.314  # J/(mol·K)
STANDARD_ATMOSPHERE_PRESSURE = 101325.0  # Pa


class PerformancesLiquidHydrogenTankTemperature(om.ExplicitComponent):
    """
    Computation of the amount of the amount of hydrogen remaining inside the tank.
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
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":tank_pressure",
            units="Pa",
            val=np.nan,
            desc="Inner thank pressure ",
        )

        self.add_output(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":liquid_hydrogen_temperature",
            units="K",
            val=20.0,
            desc="Liquid hydrogen temperature in the tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        outputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":liquid_hydrogen_temperature"
        ] = (
            1 / 20
            - np.log(
                inputs[
                    "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                    + cryogenic_hydrogen_tank_id
                    + ":tank_pressure"
                ]
                / STANDARD_ATMOSPHERE_PRESSURE
            )
            * GAS_CONSTANT
            / LIQUID_HYDROGEN_LATENT_HEAT_OF_VAPORIZATION
        ) ** -1

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        latent_heat_constant = GAS_CONSTANT / LIQUID_HYDROGEN_LATENT_HEAT_OF_VAPORIZATION

        partials[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":liquid_hydrogen_temperature",
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":tank_pressure",
        ] = latent_heat_constant / (
            inputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":tank_pressure"
            ]
            * (
                1 / 20
                - np.log(
                    inputs[
                        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                        + cryogenic_hydrogen_tank_id
                        + ":tank_pressure"
                    ]
                    / STANDARD_ATMOSPHERE_PRESSURE
                )
                * latent_heat_constant
            )
            ** 2
        )
