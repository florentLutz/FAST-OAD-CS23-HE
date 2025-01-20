# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION

STEFAN_BOLTZMANN_CONSTANT = 5.67 * 10**-8  # W/m^2.K^4
SOLAR_HEAT_FLUX = 1420.0  # W/m^2


class PerformancesCryogenicHydrogenTankRadiation(om.ExplicitComponent):
    """
    Computation of the heat radiation at the outer surface of the cryogenic tank
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

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
            desc="Option to give the position of the hydrogen gas tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]

        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        self.add_input(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_emissivity",
            val=np.nan,
            desc="Thermal emissivity of insulation material",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:length",
            val=np.nan,
            units="m",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the hydrogen gas tank",
        )

        self.add_input(
            "exterior_temperature",
            units="K",
            val=np.full(number_of_points, np.nan),
            desc="free stream temperature at the tank exterior",
        )

        self.add_input(
            "skin_temperature",
            units="K",
            val=np.full(number_of_points, np.nan),
            desc="skin temperature of the tank exterior",
        )

        self.add_input(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:reflectivity_coefficient",
            val=np.nan,
            desc="The reflectiveness of the material, values between 0 and 1, 0 means no reflectiveness",
        )

        self.add_output(
            "heat_radiation",
            units="W",
            val=np.full(number_of_points, 7.011),
            desc="heat transfer from radiation",
        )

        self.declare_partials(
            of="heat_radiation",
            wrt=["exterior_temperature", "skin_temperature"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        if position == "underbelly" or position == "wing_pod":
            self.declare_partials(
                of="heat_radiation",
                wrt=[
                    input_prefix + ":insulation:thermal_emissivity",
                    input_prefix + ":dimension:outer_diameter",
                    input_prefix + ":dimension:length",
                    input_prefix + ":insulation:reflectivity_coefficient",
                ],
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

        else:
            self.declare_partials(
                of="heat_radiation",
                wrt=[
                    input_prefix + ":insulation:thermal_emissivity",
                    input_prefix + ":dimension:outer_diameter",
                    input_prefix + ":dimension:length",
                ],
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]
        d = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter"
        ]
        l = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:length"
        ]
        if l <= 0:
            area = np.pi * d**2
            area_solar = 0.25 * np.pi * d**2
        else:
            area = np.pi * d**2 + np.pi * d * l
            area_solar = 0.25 * np.pi * d**2 + d * l
        if position == "underbelly" or position == "wing_pod":
            solar_irradiation_factor = 0.06
            solar_radiation_heat = (
                SOLAR_HEAT_FLUX
                * area_solar
                * solar_irradiation_factor
                * np.ones(number_of_points)
                * (
                    1
                    - inputs[
                        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                        + cryogenic_hydrogen_tank_id
                        + ":insulation:reflectivity_coefficient"
                    ]
                )
            )

        else:
            solar_radiation_heat = np.zeros(number_of_points)
        outputs["heat_radiation"] = (
            inputs[
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":insulation:thermal_emissivity"
            ]
            * STEFAN_BOLTZMANN_CONSTANT
            * area
            * (inputs["exterior_temperature"] ** 4 - inputs["skin_temperature"] ** 4)
            + solar_radiation_heat
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        position = self.options["position"]
        number_of_points = self.options["number_of_points"]
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )
        d = inputs[input_prefix + ":dimension:outer_diameter"]
        l = inputs[input_prefix + ":dimension:length"]
        if l <= 0:
            area = np.pi * d**2
            area_solar = 0.25 * np.pi * d**2
        else:
            area = np.pi * d**2 + np.pi * d * l
            area_solar = 0.25 * np.pi * d**2 + d * l
        r = inputs[input_prefix + ":insulation:reflectivity_coefficient"]

        partials["heat_radiation", input_prefix + ":insulation:thermal_emissivity"] = (
            STEFAN_BOLTZMANN_CONSTANT
            * area
            * (inputs["exterior_temperature"] ** 4 - inputs["skin_temperature"] ** 4)
        )
        partials["heat_radiation", "exterior_temperature"] = (
            4
            * inputs[input_prefix + ":insulation:thermal_emissivity"]
            * STEFAN_BOLTZMANN_CONSTANT
            * area
            * inputs["exterior_temperature"] ** 3
        )
        partials["heat_radiation", "skin_temperature"] = (
            -4
            * inputs[input_prefix + ":insulation:thermal_emissivity"]
            * STEFAN_BOLTZMANN_CONSTANT
            * area
            * inputs["skin_temperature"] ** 3
        )

        if position == "underbelly" or position == "wing_pod":
            solar_irradiation_factor = 0.06
            partials["heat_radiation", input_prefix + ":insulation:reflectivity_coefficient"] = (
                -area_solar * solar_irradiation_factor * SOLAR_HEAT_FLUX * np.ones(number_of_points)
            )
            if l <= 0:
                partials["heat_radiation", input_prefix + ":dimension:length"] = np.zeros(
                    number_of_points
                )
                partials["heat_radiation", input_prefix + ":dimension:outer_diameter"] = (
                    2 * np.pi * d
                ) * inputs[
                    input_prefix + ":insulation:thermal_emissivity"
                ] * STEFAN_BOLTZMANN_CONSTANT * (
                    inputs["exterior_temperature"] ** 4 - inputs["skin_temperature"] ** 4
                ) + (np.pi * d / 2) * SOLAR_HEAT_FLUX * np.ones(number_of_points) * (
                    1 - r
                ) * solar_irradiation_factor
            else:
                partials["heat_radiation", input_prefix + ":dimension:length"] = np.pi * d * inputs[
                    input_prefix + ":insulation:thermal_emissivity"
                ] * STEFAN_BOLTZMANN_CONSTANT * (
                    inputs["exterior_temperature"] ** 4 - inputs["skin_temperature"] ** 4
                ) + SOLAR_HEAT_FLUX * d * solar_irradiation_factor * np.ones(number_of_points) * (
                    1 - r
                )

                partials["heat_radiation", input_prefix + ":dimension:outer_diameter"] = (
                    2 * np.pi * d + np.pi * l
                ) * inputs[
                    input_prefix + ":insulation:thermal_emissivity"
                ] * STEFAN_BOLTZMANN_CONSTANT * (
                    inputs["exterior_temperature"] ** 4 - inputs["skin_temperature"] ** 4
                ) + (np.pi * d / 2 + l) * SOLAR_HEAT_FLUX * np.ones(number_of_points) * (
                    1 - r
                ) * solar_irradiation_factor

        else:
            if l <= 0:
                partials["heat_radiation", input_prefix + ":dimension:length"] = np.zeros(
                    number_of_points
                )

                partials["heat_radiation", input_prefix + ":dimension:outer_diameter"] = (
                    (2 * np.pi * d)
                    * inputs[input_prefix + ":insulation:thermal_emissivity"]
                    * STEFAN_BOLTZMANN_CONSTANT
                    * (inputs["exterior_temperature"] ** 4 - inputs["skin_temperature"] ** 4)
                )
            else:
                partials["heat_radiation", input_prefix + ":dimension:length"] = (
                    np.pi
                    * d
                    * inputs[input_prefix + ":insulation:thermal_emissivity"]
                    * STEFAN_BOLTZMANN_CONSTANT
                    * (inputs["exterior_temperature"] ** 4 - inputs["skin_temperature"] ** 4)
                )

                partials["heat_radiation", input_prefix + ":dimension:outer_diameter"] = (
                    (2 * np.pi * d + np.pi * l)
                    * inputs[input_prefix + ":insulation:thermal_emissivity"]
                    * STEFAN_BOLTZMANN_CONSTANT
                    * (inputs["exterior_temperature"] ** 4 - inputs["skin_temperature"] ** 4)
                )
