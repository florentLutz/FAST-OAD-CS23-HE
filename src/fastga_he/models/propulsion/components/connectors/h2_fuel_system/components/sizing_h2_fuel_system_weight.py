# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingH2FuelSystemWeight(om.ExplicitComponent):
    """
    Computation of the hydrogen fuel system weight.
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
            name="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            units="m",
            val=np.nan,
            desc="Inner diameter of the hydrogen fuel system",
        )

        self.add_input(
            name="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter",
            units="m",
            val=np.nan,
            desc="Pipe diameter of the hydrogen fuel system",
        )

        self.add_input(
            name="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_diameter",
            units="m",
            val=np.nan,
            desc="Overall diameter of the hydrogen fuel system",
        )

        self.add_input(
            name="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:length",
            units="m",
            val=np.nan,
            desc="Total length of the h2 fuel system",
        )

        self.add_input(
            name="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:density",
            val=np.nan,
            units="kg/m**3",
            desc="pipe wall material yield stress. Some reference (in kg/m^3):"
            "Steel(306):7700, Aluminum(5083):2660, ",
        )

        self.add_input(
            name="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:insulation_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_output(
            name="data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":mass",
            units="kg",
            val=10.0,
            desc="Weight of the hydrogen fuel system",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        inner_d = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter"
        ]
        pipe_d = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter"
        ]
        overall_d = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_diameter"
        ]
        length = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:length"
        ]
        pipe_dens = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:density"
        ]
        insulation_dens = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:insulation_density"
        ]

        outputs["data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":mass"] = (
            0.25
            * length
            * np.pi
            * (
                pipe_dens * (pipe_d**2.0 - inner_d**2.0)
                + insulation_dens * (overall_d**2.0 - pipe_d**2.0)
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        inner_d = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter"
        ]
        pipe_d = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter"
        ]
        overall_d = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_diameter"
        ]
        length = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:length"
        ]
        pipe_dens = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:density"
        ]
        insulation_dens = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:insulation_density"
        ]

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":mass",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:length",
        ] = (
            0.25
            * np.pi
            * (
                pipe_dens * (pipe_d**2.0 - inner_d**2.0)
                + insulation_dens * (overall_d**2.0 - pipe_d**2.0)
            )
        )

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":mass",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:density",
        ] = 0.25 * length * np.pi * (pipe_d**2.0 - inner_d**2.0)

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":mass",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:insulation_density",
        ] = 0.25 * length * np.pi * (overall_d**2.0 - pipe_d**2.0)

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":mass",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
        ] = -0.5 * length * np.pi * pipe_dens * inner_d

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":mass",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter",
        ] = 0.5 * length * np.pi * (pipe_dens * pipe_d - insulation_dens * pipe_d)

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":mass",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_diameter",
        ] = 0.5 * length * np.pi * insulation_dens * overall_d
