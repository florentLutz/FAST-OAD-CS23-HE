# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaxBoundaryLayerThickness(om.ExplicitComponent):
    """
    Computes the turbulent boundary layer thickness and the momentum thickness at the flush_inlet.
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
            desc="Identifier of the air flush_inlet",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input(
            "dynamic_viscosity",
            shape=number_of_points,
            val=np.nan,
            units="kg/m/s",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":free_stream_distance",
            val=np.nan,
            units="m",
            desc="Distance from the leading edge to the scoop shell if on the wing, or distance "
            "from the nose to the scoop shell if on the fuselage",
        )

        self.add_output(
            "boundary_layer_thickness",
            shape=number_of_points,
            val=0.016,
            units="m",
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":free_stream_distance",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        dynamic_viscosity = inputs["dynamic_viscosity"]
        free_stream_distance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":free_stream_distance"
        ]

        outputs["boundary_layer_thickness"] = (
            0.3747
            * free_stream_distance**0.8
            / (density * true_airspeed / dynamic_viscosity) ** 0.2
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        dynamic_viscosity = inputs["dynamic_viscosity"]
        free_stream_distance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":free_stream_distance"
        ]

        partials[
            "boundary_layer_thickness",
            "density",
        ] = (
            -0.07494
            * free_stream_distance**0.8
            / (true_airspeed / dynamic_viscosity) ** 0.2
            / density**1.2
        )

        partials[
            "boundary_layer_thickness",
            "true_airspeed",
        ] = (
            -0.07494
            * free_stream_distance**0.8
            / (density / dynamic_viscosity) ** 0.2
            / true_airspeed**1.2
        )

        partials[
            "boundary_layer_thickness",
            "dynamic_viscosity",
        ] = (
            0.07494
            * free_stream_distance**0.8
            / (density * true_airspeed) ** 0.2
            / dynamic_viscosity**0.8
        )

        partials[
            "boundary_layer_thickness",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":free_stream_distance",
        ] = 0.29976 / (density * true_airspeed * free_stream_distance / dynamic_viscosity) ** 0.2
