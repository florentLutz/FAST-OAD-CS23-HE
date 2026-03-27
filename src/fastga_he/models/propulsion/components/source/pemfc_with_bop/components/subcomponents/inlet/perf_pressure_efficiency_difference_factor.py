# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPressureEfficiencyDifferenceFactor(om.ExplicitComponent):
    """
    Computation of the inlet pressure efficiency difference factor due to flow condition.
    """

    def initialize(self):
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
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("density", units="kg/m**3", val=np.zeros(number_of_points))
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input(
            "air_mass_flow",
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
        self.add_input(name="dynamic_viscosity", units="kg/m/s", val=np.nan, shape=number_of_points)

        self.add_output(
            "pressure_efficiency_difference_factor",
            val=0.6,
            units="unitless",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="pressure_efficiency_difference_factor",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="pressure_efficiency_difference_factor",
            wrt=["true_airspeed", "air_mass_flow", "density", "dynamic_viscosity"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        air_mass_flow = inputs["air_mass_flow"]
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
        dynamic_viscosity = inputs["dynamic_viscosity"]

        outputs["pressure_efficiency_difference_factor"] = (
            air_mass_flow / (density * true_airspeed * highlight_width)
            - dynamic_viscosity * throat_height
        ) / throat_height

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        air_mass_flow = inputs["air_mass_flow"]
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

        partials["pressure_efficiency_difference_factor", "density"] = -air_mass_flow / (
            density**2.0 * true_airspeed * highlight_width * throat_height
        )

        partials["pressure_efficiency_difference_factor", "true_airspeed"] = -air_mass_flow / (
            density * true_airspeed**2.0 * highlight_width * throat_height
        )

        partials["pressure_efficiency_difference_factor", "air_mass_flow"] = 1.0 / (
            density * true_airspeed * highlight_width * throat_height
        )

        partials["pressure_efficiency_difference_factor", "dynamic_viscosity"] = -np.ones_like(
            density
        )

        partials[
            "pressure_efficiency_difference_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height",
        ] = -air_mass_flow / (density * true_airspeed * highlight_width * throat_height**2.0)

        partials[
            "pressure_efficiency_difference_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":highlight_width",
        ] = -air_mass_flow / (density * true_airspeed * highlight_width**2.0 * throat_height)
