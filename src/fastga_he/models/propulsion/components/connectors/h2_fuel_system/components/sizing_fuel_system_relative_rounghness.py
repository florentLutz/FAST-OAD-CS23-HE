# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingH2FuelSystemRelativeRoughness(om.ExplicitComponent):
    """
    Computation of the inner relative roughness of the H2 fuel system. The pipe inner surface
    irregularity is obtained from: https://www.pipeflow.com/pipe-pressure-drop-calculations/pipe
    -roughness.
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
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            units="mm",
            val=np.nan,
            desc="Inner diameter of the hydrogen fuel system.",
        )

        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:surface_irregularity",
            val=0.045,
            units="mm",
            desc="Pipe inner surface irregularity",
        )

        self.add_output(
            name="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":relative_roughness",
            val=0.01,
            desc="Inner diameter of the hydrogen fuel system",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        epsilon = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:surface_irregularity"
        ]

        d_inner = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter"
        ]

        outputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":relative_roughness"
        ] = epsilon / d_inner

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        epsilon = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:surface_irregularity"
        ]

        d_inner = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter"
        ]

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":relative_roughness",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
        ] = -epsilon / d_inner**2.0

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":relative_roughness",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":material:surface_irregularity",
        ] = 1.0 / d_inner
