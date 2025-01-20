# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION


PRANDTL_NUMBER = 0.71
GRAVITY_ACCELERATION = 9.81  # m/s**2


class PerformancesCryogenicHydrogenTankRayleighNumber(om.ExplicitComponent):
    """
    Computation of the tank Exterior rayleigh number
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

    def setup(self):

        number_of_points = self.options["number_of_points"]
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        self.add_input(
            name="air_kinematic_viscosity",
            units="m**2/s",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            "tank_rayleigh_number",
            val=np.full(number_of_points, 1.1),
            desc="Tank Rayleigh number at each time step",
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
            name="exterior_temperature",
            units="K",
            val=np.full(number_of_points, np.nan),
        )

        self.add_input(
            name="skin_temperature",
            units="K",
            val=np.full(number_of_points, np.nan),
        )
        self.declare_partials(
            of="tank_rayleigh_number",
            wrt=["air_kinematic_viscosity", "exterior_temperature", "skin_temperature"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="tank_rayleigh_number",
            wrt="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        rayleigh_number = (
            GRAVITY_ACCELERATION
            * (1 - inputs["skin_temperature"] / inputs["exterior_temperature"])
            * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
            * PRANDTL_NUMBER
            / inputs["air_kinematic_viscosity"]
        )
        outputs["tank_rayleigh_number"] = rayleigh_number

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        partials["tank_rayleigh_number", "skin_temperature"] = (
            -GRAVITY_ACCELERATION
            / inputs["exterior_temperature"]
            * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
            * PRANDTL_NUMBER
            / inputs["air_kinematic_viscosity"]
        )

        partials["tank_rayleigh_number", "exterior_temperature"] = (
            GRAVITY_ACCELERATION
            * inputs["skin_temperature"]
            / inputs["exterior_temperature"] ** 2
            * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
            * PRANDTL_NUMBER
            / inputs["air_kinematic_viscosity"]
        )

        partials["tank_rayleigh_number", "air_kinematic_viscosity"] = -(
            GRAVITY_ACCELERATION
            * (1 - inputs["skin_temperature"] / inputs["exterior_temperature"])
            * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
            * PRANDTL_NUMBER
            / inputs["air_kinematic_viscosity"] ** 2
        )

        partials["tank_rayleigh_number", input_prefix + ":dimension:outer_diameter"] = (
            3
            * GRAVITY_ACCELERATION
            * (1 - inputs["skin_temperature"] / inputs["exterior_temperature"])
            * inputs[input_prefix + ":dimension:outer_diameter"] ** 2
            * PRANDTL_NUMBER
            / inputs["air_kinematic_viscosity"]
        )
