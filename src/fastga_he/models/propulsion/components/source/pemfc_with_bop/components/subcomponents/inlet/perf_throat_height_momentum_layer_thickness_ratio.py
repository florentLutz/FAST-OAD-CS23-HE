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
        self.options.declare(
            name="air_inlet_id",
            default=None,
            desc="Identifier of the air inlet",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:"
            + air_inlet_id
            + ":design_air_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:"
            + air_inlet_id
            + ":design_true_airspeed",
            units="m/s",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness",
            val=np.nan,
            units="m",
        )
        self.add_input(
            "design_air_mass_flow",
            val=np.nan,
            units="kg/s",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height_layer_thickness_ratio",
            val=0.04,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:"
            + air_inlet_id
            + ":design_air_density"
        ]
        velocity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:"
            + air_inlet_id
            + ":design_true_airspeed"
        ]
        max_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness"
        ]
        design_mass_flow_rate = inputs["design_air_mass_flow"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height_layer_thickness_ratio"
        ] = 0.831 * (design_mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:"
            + air_inlet_id
            + ":design_air_density"
        ]
        velocity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:"
            + air_inlet_id
            + ":design_true_airspeed"
        ]
        max_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness"
        ]
        design_mass_flow_rate = inputs["design_air_mass_flow"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height_layer_thickness_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness",
        ] = (
            2.0
            * 0.831
            * 0.415
            * (design_mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415
            / max_thickness
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height_layer_thickness_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:"
            + air_inlet_id
            + ":design_air_density",
        ] = (
            0.831
            * 0.415
            * (design_mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415
            / density
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height_layer_thickness_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:"
            + air_inlet_id
            + ":design_true_airspeed",
        ] = (
            0.831
            * 0.415
            * (design_mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415
            / velocity
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":throat_height_layer_thickness_ratio",
            "design_air_mass_flow",
        ] = (
            -0.831
            * 0.415
            * (design_mass_flow_rate / (density * velocity * max_thickness**2.0)) ** -0.415
            / design_mass_flow_rate
        )
