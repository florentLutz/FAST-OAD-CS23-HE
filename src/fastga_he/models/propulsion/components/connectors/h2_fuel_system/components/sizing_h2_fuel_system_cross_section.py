# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingH2FuelSystemCrossSectionDimension(om.ExplicitComponent):
    """
    Computation of the hydrogen fuel system cross-section dimensions.
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
            + ":dimension:insulation_thickness",
            units="m",
            val=np.nan,
            desc="Pipe insulation layer thickness",
        )

        self.add_output(
            name="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_wall_thickness",
            units="m",
            val=0.005,
            desc="Overall wall thickness of the hydrogen fuel system",
        )

        self.add_output(
            name="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_diameter",
            units="m",
            val=0.02,
            desc="Overall diameter of the hydrogen fuel system",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_wall_thickness",
            wrt="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter",
            val=0.5,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_wall_thickness",
            wrt="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter",
            val=-0.5,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_wall_thickness",
            wrt="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:insulation_thickness",
            val=1.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_diameter",
            wrt="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter",
            val=1.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_diameter",
            wrt="data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:insulation_thickness",
            val=2.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        pipe_d = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:pipe_diameter"
        ]
        inner_d = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:inner_diameter"
        ]
        thickness_ins = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:insulation_thickness"
        ]

        outputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_wall_thickness"
        ] = 0.5 * (pipe_d - inner_d) + thickness_ins

        outputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:overall_diameter"
        ] = pipe_d + 2.0 * thickness_ins
