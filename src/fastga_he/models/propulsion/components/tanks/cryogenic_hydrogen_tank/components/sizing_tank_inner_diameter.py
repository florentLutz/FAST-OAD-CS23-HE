# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


# To modify
class SizingCryogenicHydrogenTankInnerDiameter(om.ExplicitComponent):
    """
    Computation of the inner diameter of the tank. Using the relation of the tank pressure and the yield strength of
    the wall material
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
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:wall_diameter",
            units="m",
            val=np.nan,
            desc="wall diameter of the cryogenic hydrogen tank without insulation layer",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":tank_pressure",
            val=np.nan,
            units="Pa",
            desc="Cryogenic hydrogen tank static pressure",
        )

        self.add_input(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":Safety_factor",
            val=1.0,
            desc="Cryogenic hydrogen tank design safety factor",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":material:yield_strength",
            val=np.nan,
            units="Pa",
            desc="Cryogenic hydrogen tank material yield stress",
        )

        self.add_input(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:stress_coefficient",
            val=0.5,
            desc="Coefficient at the denominator of the stress calculation",
        )

        self.add_output(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:inner_diameter",
            units="m",
            val=1.0,
            desc="Inner diameter of the cryogenic hydrogen tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        outputs[input_prefix + ":dimension:inner_diameter"] = inputs[
            input_prefix + ":dimension:wall_diameter"
        ] / (
            1
            + inputs[input_prefix + ":dimension:stress_coefficient"]
            * inputs[input_prefix + ":tank_pressure"]
            * inputs[input_prefix + ":Safety_factor"]
            / inputs[input_prefix + ":material:yield_strength"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        tank_pressure = inputs[input_prefix + ":tank_pressure"]

        sf = inputs[input_prefix + ":Safety_factor"]

        sigma = inputs[input_prefix + ":material:yield_strength"]

        d_wall = inputs[input_prefix + ":dimension:wall_diameter"]

        c = inputs[input_prefix + ":dimension:stress_coefficient"]

        partials[
            input_prefix + ":dimension:inner_diameter",
            input_prefix + ":dimension:wall_diameter",
        ] = 1 / (1 + c * tank_pressure * sf / sigma)

        partials[
            input_prefix + ":dimension:inner_diameter",
            input_prefix + ":tank_pressure",
        ] = -d_wall * sf * c * sigma / (sf * tank_pressure * c + sigma) ** 2

        partials[
            input_prefix + ":dimension:inner_diameter",
            input_prefix + ":Safety_factor",
        ] = -d_wall * tank_pressure * c * sigma / (tank_pressure * sf * c + sigma) ** 2

        partials[
            input_prefix + ":dimension:inner_diameter",
            input_prefix + ":dimension:stress_coefficient",
        ] = -d_wall * tank_pressure * sf * sigma / (tank_pressure * sf * c + sigma) ** 2

        partials[
            input_prefix + ":dimension:inner_diameter",
            input_prefix + ":material:yield_strength",
        ] = c * d_wall * sf * tank_pressure / (sigma + sf * tank_pressure * c) ** 2
