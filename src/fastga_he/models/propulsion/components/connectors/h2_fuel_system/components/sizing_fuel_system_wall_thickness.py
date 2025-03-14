# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingH2FuelSystemPipeWallThickness(om.ExplicitComponent):
    """
    Computation of the pipe wall thickness between the inner and pipe diameter.
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

        self.add_output(
            name="data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_wall_thickness",
            units="m",
            val=0.001,
            desc="Inner diameter of the hydrogen fuel system",
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter",
            val=0.5,
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            val=-0.5,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        outputs[
            "data:propulsion:he_power_train:h2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_wall_thickness"
        ] = 0.5 * (
            inputs[
                "data:propulsion:he_power_train:h2_fuel_system:"
                + h2_fuel_system_id
                + ":dimension:pipe_diameter"
            ]
            - inputs[
                "data:propulsion:he_power_train:h2_fuel_system:"
                + h2_fuel_system_id
                + ":dimension:inner_diameter"
            ]
        )
