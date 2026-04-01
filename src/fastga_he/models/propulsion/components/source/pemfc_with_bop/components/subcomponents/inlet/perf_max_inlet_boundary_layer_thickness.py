# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaxBoundaryLayerThickness(om.ExplicitComponent):
    """
    Computes the turbulent boundary layer thickness and the momentum thickness at the inlet.
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

        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("mach", val=np.nan, shape=number_of_points)
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
            + ":ramp_length",
            val=np.nan,
            units="m",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_boundary_layer_thickness",
            val=0.016,
            units="m",
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness",
            val=0.0016,
            units="m",
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_air_density",
            val=1.225,
            units="kg/m**3",
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_true_airspeed",
            val=100.0,
            units="m/s",
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_dynamic_viscosity",
            val=1.8e-5,
            units="kg/m/s",
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach",
            val=0.3,
            units="unitless",
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]
        number_of_points = self.options["number_of_points"]

        # Partials for max_boundary_layer_thickness and max_momentum_boundary_layer_thickness
        # with respect to vector inputs (density, true_airspeed, dynamic_viscosity, mach)
        self.declare_partials(
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":max_boundary_layer_thickness",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":max_momentum_boundary_layer_thickness",
            ],
            ["density", "true_airspeed", "dynamic_viscosity", "mach"],
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        # Partials for max_boundary_layer_thickness and max_momentum_boundary_layer_thickness
        # with respect to scalar input (ramp_length)
        self.declare_partials(
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":max_boundary_layer_thickness",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":max_momentum_boundary_layer_thickness",
            ],
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_length",
            method="exact",
        )
        # Partials for design outputs with respect to ramp_length (no dependency)
        self.declare_partials(
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":design_air_density",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":design_true_airspeed",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":design_dynamic_viscosity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":design_mach",
            ],
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_length",
            method="exact",
            val=0.0,
        )
        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_air_density",
            "density",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_true_airspeed",
            "true_airspeed",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_dynamic_viscosity",
            "dynamic_viscosity",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach",
            "mach",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        dynamic_viscosity = inputs["dynamic_viscosity"]
        mach = inputs["mach"]
        ramp_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_length"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_boundary_layer_thickness"
        ] = np.max(0.3747 * ramp_length**0.8 / (density * true_airspeed / dynamic_viscosity) ** 0.2)
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness"
        ] = (
            np.max(0.3747 * ramp_length**0.8 / (density * true_airspeed / dynamic_viscosity) ** 0.2)
            / 10.0
        )
        idx = np.argmax(
            0.3747 * ramp_length**0.8 / (density * true_airspeed / dynamic_viscosity) ** 0.2
        )
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_air_density"
        ] = density[idx]
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_true_airspeed"
        ] = true_airspeed[idx]
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_dynamic_viscosity"
        ] = dynamic_viscosity[idx]
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach"
        ] = mach[idx]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        dynamic_viscosity = inputs["dynamic_viscosity"]
        ramp_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_length"
        ]

        layer_thickness = (
            0.3747 * ramp_length**0.8 / (density * true_airspeed / dynamic_viscosity) ** 0.2
        )
        max_layer_thickness = np.max(layer_thickness)
        idx = np.argmax(layer_thickness)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_boundary_layer_thickness",
            "density",
        ] = np.where(
            max_layer_thickness == layer_thickness,
            -0.07494 * ramp_length**0.8 / (true_airspeed / dynamic_viscosity) ** 0.2 / density**1.2,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_boundary_layer_thickness",
            "true_airspeed",
        ] = np.where(
            max_layer_thickness == layer_thickness,
            -0.07494 * ramp_length**0.8 / (density / dynamic_viscosity) ** 0.2 / true_airspeed**1.2,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_boundary_layer_thickness",
            "dynamic_viscosity",
        ] = np.where(
            max_layer_thickness == layer_thickness,
            0.07494 * ramp_length**0.8 / (density * true_airspeed) ** 0.2 / dynamic_viscosity**0.8,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_boundary_layer_thickness",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_length",
        ] = (
            0.29976
            / (density[idx] * true_airspeed[idx] * ramp_length / dynamic_viscosity[idx]) ** 0.2
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness",
            "density",
        ] = np.where(
            max_layer_thickness == layer_thickness,
            -0.007494
            * ramp_length**0.8
            / (true_airspeed / dynamic_viscosity) ** 0.2
            / density**1.2,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness",
            "true_airspeed",
        ] = np.where(
            max_layer_thickness == layer_thickness,
            -0.007494
            * ramp_length**0.8
            / (density / dynamic_viscosity) ** 0.2
            / true_airspeed**1.2,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness",
            "dynamic_viscosity",
        ] = np.where(
            max_layer_thickness == layer_thickness,
            0.007494 * ramp_length**0.8 / (density * true_airspeed) ** 0.2 / dynamic_viscosity**0.8,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_momentum_boundary_layer_thickness",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":ramp_length",
        ] = (
            0.029975
            / (density[idx] * true_airspeed[idx] * ramp_length / dynamic_viscosity[idx]) ** 0.2
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_air_density",
            "density",
        ] = np.where(max_layer_thickness == layer_thickness, 1.0, 1e-6)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_true_airspeed",
            "true_airspeed",
        ] = np.where(max_layer_thickness == layer_thickness, 1.0, 1e-6)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_dynamic_viscosity",
            "dynamic_viscosity",
        ] = np.where(max_layer_thickness == layer_thickness, 1.0, 1e-6)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach",
            "mach",
        ] = np.where(max_layer_thickness == layer_thickness, 1.0, 1e-6)
