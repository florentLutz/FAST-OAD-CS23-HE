# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesThroatHeightMomentumBoundaryLayerThicknessRatio(om.ExplicitComponent):
    """
    Computation of the throat height layer thickness ratio.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input("design_air_density", units="kg/m**3", val=np.nan)
        self.add_input("design_true_airspeed", units="m/s", val=np.nan)
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:max_momentum_boundary_layer_thickness",
            val=1e-5,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            units="kg/s",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:throat_height_layer_thickness_ratio",
            val=1e-4,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        density = inputs["design_air_density"]
        velocity = inputs["design_true_airspeed"]
        max_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:max_momentum_boundary_layer_thickness"
        ]
        mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:throat_height_layer_thickness_ratio"
        ] = 0.831 * (mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        density = inputs["design_air_density"]
        velocity = inputs["design_true_airspeed"]
        max_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:max_momentum_boundary_layer_thickness"
        ]
        mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:throat_height_layer_thickness_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:max_momentum_boundary_layer_thickness",
        ] = (
            2.0
            * 0.831
            * 0.415
            * (mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415
            / max_thickness
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:throat_height_layer_thickness_ratio",
            "design_air_density",
        ] = (
            0.831
            * 0.415
            * (mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415
            / density
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:throat_height_layer_thickness_ratio",
            "design_true_airspeed",
        ] = (
            0.831
            * 0.415
            * (mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415
            / velocity
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":inlet:throat_height_layer_thickness_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
        ] = (
            -0.831
            * 0.415
            * (mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415
            / mass_flow_rate
        )
