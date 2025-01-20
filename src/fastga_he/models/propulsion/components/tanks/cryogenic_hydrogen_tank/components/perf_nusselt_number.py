# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION


PRANDTL_NUMBER = 0.71
GRAVITY_ACCELERATION = 9.81  # m/s**2


class PerformancesCryogenicHydrogenTankNusseltNumber(om.ExplicitComponent):
    """
    Computation of the amount of the amount of hydrogen boil-off during the mission.
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

        self.add_output(
            "tank_nusselt_number",
            val=np.full(number_of_points, 5.272),
            desc="Tank Nusselt number at each time step",
        )
        self.add_input(
            name="true_airspeed",
            units="m/s",
            val=np.full(number_of_points, np.nan),
        )

        if position == "wing_pod" or position == "underbelly":
            self.add_input(
                name="air_kinematic_viscosity",
                units="m**2/s",
                val=np.full(number_of_points, np.nan),
            )
            self.add_input(
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter",
                val=np.nan,
                units="m",
                desc="Outer diameter of the hydrogen gas tank",
            )
            self.declare_partials(
                of="tank_nusselt_number",
                wrt=["true_airspeed", "air_kinematic_viscosity"],
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )
            self.declare_partials(
                of="tank_nusselt_number",
                wrt="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

        else:
            self.add_input(
                "tank_rayleigh_number",
                val=np.full(number_of_points, 1.1),
                desc="Tank Rayleigh number at each time step",
            )

            self.add_input(
                name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:aspect_ratio",
                val=2.0,
                desc="Tank aspect between the overall length and outer diameter, the higher the more cylindrical",
            )

            self.declare_partials(
                of="tank_nusselt_number",
                wrt=["tank_rayleigh_number", "true_airspeed"],
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )

            self.declare_partials(
                of="tank_nusselt_number",
                wrt="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:aspect_ratio",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )
        air_speed = inputs["true_airspeed"]

        if position == "wing_pod" or position == "underbelly":
            reynolds_number = (
                air_speed
                * inputs[input_prefix + ":dimension:outer_diameter"]
                / inputs["air_kinematic_viscosity"]
            )
            outputs["tank_nusselt_number"] = (
                0.03625 * PRANDTL_NUMBER ** 0.43 * reynolds_number ** 0.8
            )
        else:
            rayleigh_number = inputs["tank_rayleigh_number"]
            ar = inputs[input_prefix + ":dimension:aspect_ratio"]
            outputs["tank_nusselt_number"] = (0.06 + 0.3213 * rayleigh_number ** (1 / 6)) ** 2 * (
                1 - 1 / ar
            ) + (2 + 0.4545 * rayleigh_number ** 0.25) / ar

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        if position == "wing_pod" or position == "underbelly":

            partials["tank_nusselt_number", "true_airspeed"] = (
                0.03625
                * PRANDTL_NUMBER ** 0.43
                * 0.8
                * (
                    inputs[input_prefix + ":dimension:outer_diameter"]
                    / inputs["air_kinematic_viscosity"]
                )
                ** 0.8
                / inputs["true_airspeed"] ** 0.2
            )

            partials["tank_nusselt_number", input_prefix + ":dimension:outer_diameter"] = (
                0.03625
                * PRANDTL_NUMBER ** 0.43
                * 0.8
                * (inputs["true_airspeed"] / inputs["air_kinematic_viscosity"]) ** 0.8
                / inputs[input_prefix + ":dimension:outer_diameter"] ** 0.2
            )

            partials["tank_nusselt_number", "air_kinematic_viscosity"] = (
                -0.03625
                * PRANDTL_NUMBER ** 0.43
                * 0.8
                * (inputs["true_airspeed"] * inputs[input_prefix + ":dimension:outer_diameter"])
                ** 0.8
                / inputs["air_kinematic_viscosity"] ** 1.8
            )

        else:
            rayleigh_number = inputs["tank_rayleigh_number"]
            ar = inputs[input_prefix + ":dimension:aspect_ratio"]
            partials["tank_nusselt_number", "tank_rayleigh_number"] = (
                0.3213
                * (0.3213 * rayleigh_number ** (1 / 6) + 0.06)
                / 3
                / rayleigh_number ** (5 / 6)
                * (1 - 1 / ar)
                + 0.4545 / 4 / rayleigh_number ** 0.75 / ar
            )

            partials["tank_nusselt_number", input_prefix + ":dimension:aspect_ratio"] = (
                0.06 + 0.3213 * rayleigh_number ** (1 / 6)
            ) ** 2 / ar ** 2 - (2 + 0.4545 * rayleigh_number ** 0.25) / ar ** 2

            partials["tank_nusselt_number", "true_airspeed"] = np.zeros_like(rayleigh_number)
