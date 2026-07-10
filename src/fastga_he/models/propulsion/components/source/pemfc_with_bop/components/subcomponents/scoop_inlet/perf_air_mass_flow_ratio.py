# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from fastga_he.models.propulsion.components.source.pemfc_with_bop.components.subcomponents.fluid_characteristics import (
    FluidSpecificHeatCapacity,
)


class PerformancesAirInletAirMassFlow(om.ExplicitComponent):
    """
    Compute air mass flow ratio for scoop inlet sizing.
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
            desc="Identifier of the air_inlet",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_input(
            "air_consumption",
            units="kg/s",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass_flow_factor",
            val=3.0,
            units="unitless",
        )

        self.add_output("inlet_air_mass_flow", val=1.8, units="kg/s", shape=number_of_points)
        self.add_output("total_air_mass_flow", val=2.1, units="kg/s", shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials("*", "*", method="exact")
        self.declare_partials(
            "*",
            "air_consumption",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        outputs["inlet_air_mass_flow"] = np.clip(
            inputs["air_consumption"]
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":mass_flow_factor"
            ],
            0.001,
            np.inf,
        )
        outputs["total_air_mass_flow"] = np.clip(
            inputs["air_consumption"]
            * (
                inputs[
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + air_inlet_id
                    + ":mass_flow_factor"
                ]
                + 1.0
            ),
            0.001,
            np.inf,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]
        number_of_points = self.options["number_of_points"]

        unclipped_air_mass_flow = (
            inputs["air_consumption"]
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":mass_flow_factor"
            ]
        )
        inclipped_total_air_mass_flow = inputs["air_consumption"] * (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":mass_flow_factor"
            ]
            + 1.0
        )

        clipped_air_mass_flow = np.clip(unclipped_air_mass_flow, 0.001, np.inf)
        clipped_total_air_mass_flow = np.clip(inclipped_total_air_mass_flow, 0.001, np.inf)

        partials["inlet_air_mass_flow", "air_consumption"] = np.where(
            unclipped_air_mass_flow == clipped_air_mass_flow,
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":mass_flow_factor"
            ]
            * np.ones(number_of_points),
            1e-6,
        )

        partials[
            "inlet_air_mass_flow",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass_flow_factor",
        ] = np.where(
            unclipped_air_mass_flow == clipped_air_mass_flow, inputs["air_consumption"], 1e-6
        )

        partials["total_air_mass_flow", "air_consumption"] = np.where(
            inclipped_total_air_mass_flow == clipped_total_air_mass_flow,
            (
                inputs[
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + air_inlet_id
                    + ":mass_flow_factor"
                ]
                + 1.0
            )
            * np.ones(number_of_points),
            1e-6,
        )

        partials[
            "total_air_mass_flow",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass_flow_factor",
        ] = np.where(
            inclipped_total_air_mass_flow == clipped_total_air_mass_flow,
            inputs["air_consumption"],
            1e-6,
        )


class _CaptureAreaGuess(om.ExplicitComponent):
    """
    Compute the capture area guess for scoop inlet sizing.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("air_consumption", units="kg/s", val=np.nan, shape=number_of_points)
        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)

        self.add_output("capture_area_guess", val=1.8, units="m**2")
        self.add_output("capture_area_guess_index", val=0, units="unitless")

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            "capture_area_guess",
            "*",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            "capture_area_guess_index",
            "*",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=0.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        air_consumption = inputs["air_consumption"]
        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]

        outputs["capture_area_guess"] = np.max(air_consumption / (density * true_airspeed))
        outputs["capture_area_guess_index"] = np.argmax(air_consumption / (density * true_airspeed))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        air_consumption = inputs["air_consumption"]
        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]

        capture_area_guess = air_consumption / (density * true_airspeed)
        max_capture_area_guess = np.max(capture_area_guess)

        partials["capture_area_guess", "air_consumption"] = np.where(
            capture_area_guess == max_capture_area_guess, 1.0 / (density * true_airspeed), 1e-6
        )

        partials["capture_area_guess", "density"] = np.where(
            capture_area_guess == max_capture_area_guess,
            -air_consumption / (density**2.0 * true_airspeed),
            1e-6,
        )

        partials["capture_area_guess", "true_airspeed"] = np.where(
            capture_area_guess == max_capture_area_guess,
            -air_consumption / (density * true_airspeed**2.0),
            1e-6,
        )


class _upperdistanceguess(om.ExplicitComponent):
    """
    Compute air mass flow ratio for scoop inlet sizing.
    """

    def initialize(self):
        self.options.declare(
            "inlet_shape", default="circular", desc=" shape of the inlet (circular or semicircular)"
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
            desc="Identifier of the air_inlet",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_input(
            "capture_area_guess",
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lower_distance",
            val=np.nan,
            units="m",
        )

        self.add_output("upper_distance_guess", val=0.15, units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inlet_shape = self.options["inlet_shape"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        capture_area_guess = inputs["capture_area_guess"]
        lower_distance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lower_distance"
        ]

        if inlet_shape == "circular":
            outputs["upper_distance_guess"] = np.sqrt(4.0 * capture_area_guess / np.pi) + (
                lower_distance
            )
        elif inlet_shape == "semicircular":
            outputs["upper_distance_guess"] = (
                np.sqrt(8.0 * capture_area_guess / np.pi) + lower_distance
            )
        else:
            raise ValueError("Invalid inlet shape. Must be 'circular' or 'semicircular'.")

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inlet_shape = self.options["inlet_shape"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        capture_area_guess = inputs["capture_area_guess"]

        partials["upper_distance_guess", "capture_area_guess"] = (
            4.0 / (2.0 * np.sqrt(np.pi) * np.sqrt(4.0 * capture_area_guess))
            if inlet_shape == "circular"
            else 8.0 / (2.0 * np.sqrt(np.pi) * np.sqrt(8.0 * capture_area_guess))
        )

        partials[
            "upper_distance_guess",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":lower_distance",
        ] = 1.0


class _MachNumberGuess(om.ExplicitComponent):
    """
    Compute air mass flow ratio for scoop inlet sizing.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("mach", units="unitless", val=np.nan, shape=number_of_points)
        self.add_input("capture_area_guess_index", units="unitless", val=np.nan)

        self.add_output("mach_number_design", val=0.15, units="unitless")

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            "mach_number_design",
            "mach",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            "mach_number_design",
            "capture_area_guess_index",
            method="exact",
            val=0.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mach = inputs["mach"]
        capture_area_guess_index = int(inputs["capture_area_guess_index"])

        outputs["mach_number_design"] = mach[capture_area_guess_index]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mach = inputs["mach"]
        capture_area_guess_index = int(inputs["capture_area_guess_index"])

        partials["mach_number_design", "mach"] = np.where(
            np.arange(mach.size) == capture_area_guess_index,
            1.0,
            1e-6,
        )


class _DesignBoundaryLayerThickness(om.ExplicitComponent):
    """
    Compute air mass flow ratio for scoop inlet sizing.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("capture_area_guess_index", units="unitless", val=np.nan)
        self.add_input("boundary_layer_thickness", units="m", val=np.nan, shape=number_of_points)

        self.add_output("design_boundary_layer_thickness", val=0.016, units="m")

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            "design_boundary_layer_thickness",
            "boundary_layer_thickness",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            "design_boundary_layer_thickness",
            "capture_area_guess_index",
            method="exact",
            val=0.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        capture_area_guess_index = int(inputs["capture_area_guess_index"])
        boundary_layer_thickness = inputs["boundary_layer_thickness"]

        outputs["design_boundary_layer_thickness"] = boundary_layer_thickness[
            capture_area_guess_index
        ]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        capture_area_guess_index = int(inputs["capture_area_guess_index"])
        boundary_layer_thickness = inputs["boundary_layer_thickness"]

        partials["design_boundary_layer_thickness", "boundary_layer_thickness"] = np.where(
            np.arange(boundary_layer_thickness.size) == capture_area_guess_index,
            1.0,
            1e-6,
        )


class _MachFactorMassFlowRatio(om.ExplicitComponent):
    """
    Compute air mass flow ratio for scoop inlet sizing.
    """

    def setup(self):
        self.add_input("mach_number_design", units="unitless", val=np.nan)

        self.add_output("mach_factor_mass_flow_ratio", val=6.6, units="unitless")

    def setup_partials(self):
        self.declare_partials(
            "mach_factor_mass_flow_ratio",
            "*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["mach_factor_mass_flow_ratio"] = (
            7.09
            - 0.814 * inputs["mach_number_design"]
            - 0.792 * inputs["mach_number_design"] ** 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["mach_factor_mass_flow_ratio", "mach_number_design"] = (
            -1.584 * inputs["mach_number_design"] - 0.814
        )


class _HeightBoundaryLayerThicknessRatio(om.ExplicitComponent):
    """
    Compute air mass flow ratio for scoop inlet sizing.
    """

    def setup(self):
        self.add_input("height", units="m", val=np.nan)
        self.add_input("design_boundary_layer_thickness", units="m", val=np.nan)

        self.add_output("height_boundary_layer_thickness_ratio", val=2.0, units="unitless")

    def setup_partials(self):
        self.declare_partials(
            "height_boundary_layer_thickness_ratio",
            "*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["height_boundary_layer_thickness_ratio"] = (
            inputs["height"] / inputs["design_boundary_layer_thickness"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        height = inputs["height"]
        boundary_layer_thickness = inputs["design_boundary_layer_thickness"]

        partials["height_boundary_layer_thickness_ratio", "height"] = 1.0 / boundary_layer_thickness

        partials["height_boundary_layer_thickness_ratio", "design_boundary_layer_thickness"] = (
            -height / (boundary_layer_thickness**2.0)
        )


class _MassRatio(om.ExplicitComponent):
    """
    Compute air mass flow ratio for scoop inlet sizing.
    """

    def setup(self):
        self.add_input("mach_factor_mass_flow_ratio", units="unitless", val=np.nan)
        self.add_input("height_boundary_layer_thickness_ratio", units="unitless", val=np.nan)

        self.add_output("mass_flow_ratio", val=6.6, units="unitless")

    def setup_partials(self):
        self.declare_partials(
            "mass_flow_ratio",
            "*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        height_boundary_layer_thickness_ratio = inputs["height_boundary_layer_thickness_ratio"]
        mach_factor_mass_flow_ratio = inputs["mach_factor_mass_flow_ratio"]

        if height_boundary_layer_thickness_ratio < 1.0:
            outputs["mass_flow_ratio"] = 1.0 - height_boundary_layer_thickness_ratio / (
                mach_factor_mass_flow_ratio + 1.0
            )
        else:
            outputs["mass_flow_ratio"] = (
                mach_factor_mass_flow_ratio
                / (mach_factor_mass_flow_ratio + 1.0)
                * height_boundary_layer_thickness_ratio ** (-1.0 / mach_factor_mass_flow_ratio)
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        height_boundary_layer_thickness_ratio = inputs["height_boundary_layer_thickness_ratio"]
        mach_factor_mass_flow_ratio = inputs["mach_factor_mass_flow_ratio"]
        layer_thickness_ratio_larger_than_one = height_boundary_layer_thickness_ratio >= 1.0

        partials["mass_flow_ratio", "mach_factor_mass_flow_ratio"] = (
            (
                (np.log(height_boundary_layer_thickness_ratio) + 1.0) * mach_factor_mass_flow_ratio
                + np.log(height_boundary_layer_thickness_ratio)
            )
            / (
                height_boundary_layer_thickness_ratio ** (1.0 / mach_factor_mass_flow_ratio)
                * mach_factor_mass_flow_ratio
                * (mach_factor_mass_flow_ratio + 1.0) ** 2.0
            )
            if layer_thickness_ratio_larger_than_one
            else (
                height_boundary_layer_thickness_ratio / (mach_factor_mass_flow_ratio + 1.0) ** 2.0
            )
        )

        partials["mass_flow_ratio", "height_boundary_layer_thickness_ratio"] = (
            (-(height_boundary_layer_thickness_ratio ** (-1.0 / mach_factor_mass_flow_ratio - 1.0)))
            / (mach_factor_mass_flow_ratio + 1.0)
            if layer_thickness_ratio_larger_than_one
            else (-1.0 / (mach_factor_mass_flow_ratio + 1.0))
        )
