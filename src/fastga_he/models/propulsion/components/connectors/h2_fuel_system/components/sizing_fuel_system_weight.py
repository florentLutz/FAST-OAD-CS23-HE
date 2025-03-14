# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingFuelSystemWeight(om.ExplicitComponent):
    """
    Computation of the hydrogen fuel system weight, based on a formula from :cite:`gudmundsson:2013` for
    Torenbeek approach. Include the weight of fuel tanks, pipes, pumps, vents, ...
    """

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
            name="data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            units="m",
            val=np.nan,
            desc="Inner diameter of the hydrogen fuel system",
        )

        self.add_input(
            name="data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter",
            units="m",
            val=np.nan,
            desc="Pipe diameter of the hydrogen fuel system",
        )

        self.add_input(
            name="data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:insulation_thickness",
            units="m",
            val=np.nan,
            desc="Pipe insulation layer thickness",
        )

        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            units="m",
            val=np.nan,
            desc="Total length of the h2 fuel system",
        )

        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:density",
            val=np.nan,
            units="kg/m**3",
            desc="pipe wall material yield stress. Some reference (in kg/m^3):"
            "Steel(306):7700, Aluminum(5083):2660, ",
        )

        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:insulation_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_output(
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":mass",
            units="kg",
            val=10.0,
            desc="Weight of the hydrogen fuel system",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        inner_d = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter"
        ]
        pipe_d = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter"
        ]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        volume = inputs[
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":connected_volume"
        ]

        partials[
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":mass",
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":connected_volume",
        ] = self.factor * self.exponent * volume ** (self.exponent - 1.0)
