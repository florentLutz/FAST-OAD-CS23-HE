# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingH2FuelSystemInnerDiameter(om.ExplicitComponent):
    """
    Computation of the inner diameter of the H2 fuel system. Using the relation of the pipe
    pressure and the yield strength of the wall material :cite:`colozza:2002`.
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
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the hydrogen fuel system excluding insulation.",
        )

        self.add_input(
            "data:propulsion:he_power_train:h2_fuel_system:" + h2_fuel_system_id + ":pipe_pressure",
            val=np.nan,
            units="Pa",
            desc="hydrogen transport pressure",
        )

        self.add_input(
            name="data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":safety_factor",
            val=1.0,
            desc="hydrogen fuel system design safety factor",
        )

        self.add_input(
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":material:yield_strength",
            val=np.nan,
            units="Pa",
            desc="pipe wall material yield stress. Some reference (in MPa):"
            "Steel(306):240, Aluminum(5083):228, ",
        )

        self.add_output(
            name="data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            units="m",
            val=0.01,
            desc="Inner diameter of the hydrogen fuel system",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        safety_factor = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:" + h2_fuel_system_id + ":safety_factor"
        ]

        sigma = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":material:yield_strength"
        ]

        d_pipe = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter"
        ]

        pipe_pressure = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:" + h2_fuel_system_id + ":pipe_pressure"
        ]

        outputs[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter"
        ] = 2.0 * d_pipe * sigma / (pipe_pressure * safety_factor + 2.0 * sigma)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        pipe_pressure = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:" + h2_fuel_system_id + ":pipe_pressure"
        ]

        safety_factor = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:" + h2_fuel_system_id + ":safety_factor"
        ]

        sigma = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":material:yield_strength"
        ]

        d_pipe = inputs[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter"
        ]

        partials[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter",
        ] = 2.0 * sigma / (2.0 * sigma + pipe_pressure * safety_factor)

        partials[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            "data:propulsion:he_power_train:h2_fuel_system:" + h2_fuel_system_id + ":pipe_pressure",
        ] = (
            -2.0
            * d_pipe
            * safety_factor
            * sigma
            / (safety_factor * pipe_pressure + 2.0 * sigma) ** 2.0
        )

        partials[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            "data:propulsion:he_power_train:h2_fuel_system:" + h2_fuel_system_id + ":safety_factor",
        ] = (
            -2.0
            * d_pipe
            * pipe_pressure
            * sigma
            / (pipe_pressure * safety_factor + 2.0 * sigma) ** 2.0
        )

        partials[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":material:yield_strength",
        ] = 2.0 * (
            d_pipe
            * safety_factor
            * pipe_pressure
            / (2.0 * sigma + safety_factor * pipe_pressure) ** 2.0
        )
