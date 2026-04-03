# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesThroatAirSpeed(om.ExplicitComponent):
    """
    Computation of the inlet throat airspeed.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="air_inlet_id",
            default=None,
            desc="Identifier of the air inlet",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_input("density", units="kg/m**3", val=np.zeros(number_of_points))
        self.add_input(
            "inlet_air_mass_flow",
            val=np.nan,
            units="kg/s",
            shape=number_of_points,
        )

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height",
            val=np.nan,
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":highlight_width",
            val=np.nan,
            units="m",
        )

        self.add_output(
            "throat_air_speed",
            val=120.0,
            units="m/s",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt=["density", "inlet_air_mass_flow"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs["density"]
        air_mass_flow = inputs["inlet_air_mass_flow"]
        throat_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height"
        ]
        highlight_width = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":highlight_width"
        ]

        outputs["throat_air_speed"] = air_mass_flow / (density * throat_height * highlight_width)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs["density"]
        air_mass_flow = inputs["inlet_air_mass_flow"]
        throat_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height"
        ]
        highlight_width = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":highlight_width"
        ]

        partials["throat_air_speed", "density"] = -air_mass_flow / (
            density**2.0 * throat_height * highlight_width
        )

        partials["throat_air_speed", "inlet_air_mass_flow"] = 1.0 / (
            density * throat_height * highlight_width
        )

        partials[
            "throat_air_speed",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height",
        ] = -air_mass_flow / (density * throat_height**2.0 * highlight_width)

        partials[
            "throat_air_speed",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":highlight_width",
        ] = -air_mass_flow / (density * throat_height * highlight_width**2.0)
