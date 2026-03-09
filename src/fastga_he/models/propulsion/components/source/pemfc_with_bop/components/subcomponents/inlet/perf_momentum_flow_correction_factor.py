# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMomentumFlowCorrectionFactor(om.Group):
    """
    Computation of the momentum flow correction factor of a flush inlet. This is part of the inlet
    ram drag computation.
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

        self.add_subsystem(
            "log10_design_mach",
            _Log10(),
            promotes=[
                (
                    "x",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + air_inlet_id
                    + ":design_mach",
                ),
                ("log10_x", "log10_design_mach"),
            ],
        )
        self.add_subsystem(
            "log10_layer_thickness_highlight_height_ratio",
            _Log10(),
            promotes=[
                (
                    "x",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + air_inlet_id
                    + ":layer_thickness_highlight_height_ratio",
                ),
                ("log10_x", "log10_layer_thickness_highlight_height_ratio"),
            ],
        )
        self.add_subsystem(
            "layer_thickness_highlight_height_ratio_power",
            _LayerThicknessHighlightHeightRatioPower(),
            promotes=["*"],
        )
        self.add_subsystem(
            "design_mach_power",
            _DesignMachPower(),
            promotes=["*"],
        )
        self.add_subsystem(
            "momentum_flow_correction_factor",
            _MomentumFlowCorrectionFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )


class _MomentumFlowCorrectionFactor(om.ExplicitComponent):
    """
    Computation of the momentum flow correction factor of a flush inlet. This is part of the inlet
    ram drag computation.
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
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":layer_thickness_highlight_height_ratio",
            val=np.nan,
            units="unitless",
        )
        self.add_input("layer_thickness_highlight_height_ratio_power", val=np.nan, units="unitless")
        self.add_input("design_mach_power", val=np.nan, units="unitless")

        self.add_output(
            "momentum_flow_correction_factor",
            val=1e-4,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach"
        ]
        design_mach_power = inputs["design_mach_power"]
        layer_thickness_highlight_height_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":layer_thickness_highlight_height_ratio"
        ]
        layer_thickness_highlight_height_ratio_power = inputs[
            "layer_thickness_highlight_height_ratio_power"
        ]

        outputs["momentum_flow_correction_factor"] = (
            10**-0.12877
            * layer_thickness_highlight_height_ratio**layer_thickness_highlight_height_ratio_power
            * design_mach**design_mach_power
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach"
        ]
        design_mach_power = inputs["design_mach_power"]
        layer_thickness_highlight_height_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":layer_thickness_highlight_height_ratio"
        ]
        layer_thickness_highlight_height_ratio_power = inputs[
            "layer_thickness_highlight_height_ratio_power"
        ]

        partials[
            "momentum_flow_correction_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach",
        ] = (
            10**-0.12877
            * layer_thickness_highlight_height_ratio**layer_thickness_highlight_height_ratio_power
            * design_mach ** (design_mach_power - 1.0)
            * design_mach_power
        )

        partials[
            "momentum_flow_correction_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":layer_thickness_highlight_height_ratio",
        ] = (
            10**-0.12877
            * layer_thickness_highlight_height_ratio
            ** (layer_thickness_highlight_height_ratio_power - 1.0)
            * design_mach**design_mach_power
            * layer_thickness_highlight_height_ratio_power
        )

        partials["momentum_flow_correction_factor", "design_mach_power"] = (
            10**-0.12877
            * layer_thickness_highlight_height_ratio**layer_thickness_highlight_height_ratio_power
            * design_mach**design_mach_power
            * np.log(design_mach)
        )

        partials[
            "momentum_flow_correction_factor", "layer_thickness_highlight_height_ratio_power"
        ] = (
            10**-0.12877
            * layer_thickness_highlight_height_ratio**layer_thickness_highlight_height_ratio_power
            * design_mach**design_mach_power
            * np.log(layer_thickness_highlight_height_ratio)
        )


class _Log10(om.ExplicitComponent):
    """
    log10 of both inputs.
    """

    def setup(self):
        self.add_input("x", val=np.nan, units="unitless")

        self.add_output("log10_x", val=-0.48, units="unitless")

    def setup_partials(self):
        self.declare_partials("log10_x", "x", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["log10_x"] = np.log10(inputs["x"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["log10_x", "x"] = 1 / (inputs["x"] * np.log(10))


class _LayerThicknessHighlightHeightRatioPower(om.ExplicitComponent):
    """
    The power term of the layer thickness to highlight height ratio in the momentum flow correction factor computation.
    """

    def setup(self):
        self.add_input("log10_layer_thickness_highlight_height_ratio", val=np.nan)
        self.add_input("log10_design_mach", val=np.nan)

        self.add_output("layer_thickness_highlight_height_ratio_power", val=-0.182)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")
        self.declare_partials(
            "layer_thickness_highlight_height_ratio_power", "log10_design_mach", val=-0.03841
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        log10_layer_thickness_highlight_height_ratio = inputs[
            "log10_layer_thickness_highlight_height_ratio"
        ]
        log10_design_mach = inputs["log10_design_mach"]

        outputs["layer_thickness_highlight_height_ratio_power"] = (
            -0.2636
            - 0.03841 * log10_design_mach
            + 0.06416 * log10_layer_thickness_highlight_height_ratio**2.0
            - 0.11447 * log10_layer_thickness_highlight_height_ratio
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "layer_thickness_highlight_height_ratio_power",
            "log10_layer_thickness_highlight_height_ratio",
        ] = 0.12832 * inputs["log10_layer_thickness_highlight_height_ratio"] - 0.11447


class _DesignMachPower(om.ExplicitComponent):
    """
    The power term of the design Mach number in the momentum flow correction factor computation.
    """

    def setup(self):
        self.add_input("log10_design_mach", val=np.nan)

        self.add_output("design_mach_power", val=-0.034)

    def setup_partials(self):
        self.declare_partials("*", "*", val=-0.0682)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["design_mach_power"] = -0.06691 - 0.0682 * inputs["log10_design_mach"]
