# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHeatSinkFinHeight(om.Group):
    """
    Computing heat sink fin height.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="finned_heat_sink_id",
            default=None,
            desc="Identifier of the finned heat sink",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        self.add_subsystem(
            "hyperbolic_tangent",
            _HyperbolicTangent(
                pemfc_stack_bop_id=pemfc_stack_bop_id, finned_heat_sink_id=finned_heat_sink_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inverse_hyperbolic_tangent",
            _InverseHyperbolicTangent(),
            promotes=["*"],
        )
        self.add_subsystem(
            "fin_height",
            _FinHeight(
                pemfc_stack_bop_id=pemfc_stack_bop_id, finned_heat_sink_id=finned_heat_sink_id
            ),
            promotes=["*"],
        )


class _HyperbolicTangent(om.ExplicitComponent):
    """
    Computing the hyperbolic tangent of a value, used for the regularization of the fin height
    sizing equation.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="finned_heat_sink_id",
            default=None,
            desc="Identifier of the finned heat sink",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":design_dissipation_power",
            units="W",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_parameter",
            units="m**-1",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_heat_transfer_parameter",
            units="W/K",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":conduction_coefficient",
            units="W/m/K",
            val=150.0,
            desc="Thermal conductivity for 6061 aluminum alloy",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":convection_heat_transfer_coefficient",
            units="W/m**2/K",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":number_of_fins",
            units="unitless",
            val=np.nan,
        )

        self.add_output(
            "hyperbolic_tangent",
            units="unitless",
            val=0.01,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        design_dissipation_power = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":design_dissipation_power"
        ]
        number_of_fins = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":number_of_fins"
        ]
        fin_parameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_parameter"
        ]
        fin_heat_transfer_parameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_heat_transfer_parameter"
        ]
        convection_heat_transfer_coefficient = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":convection_heat_transfer_coefficient"
        ]
        conduction_coefficient = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":conduction_coefficient"
        ]

        outputs["hyperbolic_tangent"] = (
            design_dissipation_power
            - number_of_fins
            * fin_heat_transfer_parameter
            * (convection_heat_transfer_coefficient / (fin_parameter * conduction_coefficient))
        ) / (
            number_of_fins * fin_heat_transfer_parameter
            - design_dissipation_power
            * (convection_heat_transfer_coefficient / (fin_parameter * conduction_coefficient))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        design_dissipation_power = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":design_dissipation_power"
        ]
        number_of_fins = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":number_of_fins"
        ]
        fin_parameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_parameter"
        ]
        fin_heat_transfer_parameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_heat_transfer_parameter"
        ]
        convection_heat_transfer_coefficient = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":convection_heat_transfer_coefficient"
        ]
        conduction_coefficient = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":conduction_coefficient"
        ]

        common_denominator = (
            convection_heat_transfer_coefficient * design_dissipation_power
            - fin_heat_transfer_parameter * number_of_fins * conduction_coefficient * fin_parameter
        ) ** 2.0

        partials[
            "hyperbolic_tangent",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":design_dissipation_power",
        ] = (
            fin_heat_transfer_parameter
            * number_of_fins
            * (
                conduction_coefficient**2.0 * fin_parameter**2.0
                - convection_heat_transfer_coefficient**2.0
            )
        ) / common_denominator

        partials[
            "hyperbolic_tangent",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":number_of_fins",
        ] = (
            -(
                fin_heat_transfer_parameter
                * (
                    conduction_coefficient**2.0 * fin_parameter**2.0
                    - convection_heat_transfer_coefficient**2.0
                )
                * design_dissipation_power
            )
            / common_denominator
        )

        partials[
            "hyperbolic_tangent",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_parameter",
        ] = -(
            (
                convection_heat_transfer_coefficient
                * conduction_coefficient
                * (
                    design_dissipation_power**2.0
                    - fin_heat_transfer_parameter**2.0 * number_of_fins**2.0
                )
            )
            / common_denominator
        )

        partials[
            "hyperbolic_tangent",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_heat_transfer_parameter",
        ] = -(
            number_of_fins
            * (
                conduction_coefficient**2.0 * fin_parameter**2.0
                - convection_heat_transfer_coefficient**2.0
            )
            * design_dissipation_power
            / common_denominator
        )

        partials[
            "hyperbolic_tangent",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":convection_heat_transfer_coefficient",
        ] = (
            conduction_coefficient
            * fin_parameter
            * (
                design_dissipation_power**2.0
                - fin_heat_transfer_parameter**2.0 * number_of_fins**2.0
            )
            / common_denominator
        )

        partials[
            "hyperbolic_tangent",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":conduction_coefficient",
        ] = (
            -convection_heat_transfer_coefficient
            * fin_parameter
            * (
                design_dissipation_power**2.0
                - fin_heat_transfer_parameter**2.0 * number_of_fins**2.0
            )
            / common_denominator
        )


class _InverseHyperbolicTangent(om.ExplicitComponent):
    """
    Computing the inverse hyperbolic tangent of a value, used for the regularization of the fin height
    sizing equation.
    """

    def setup(self):
        self.add_input(
            "hyperbolic_tangent",
            units="unitless",
            val=np.nan,
        )

        self.add_output(
            "inverse_hyperbolic_tangent",
            units="unitless",
            val=0.01,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        hyperbolic_tangent = np.clip(inputs["hyperbolic_tangent"], -0.999999, 0.999999)

        outputs["inverse_hyperbolic_tangent"] = np.arctanh(hyperbolic_tangent)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        hyperbolic_tangent = np.clip(inputs["hyperbolic_tangent"], -0.999999, 0.999999)

        partials["inverse_hyperbolic_tangent", "hyperbolic_tangent"] = (
            1.0 / (1.0 - hyperbolic_tangent**2.0)
            if -0.999999 < hyperbolic_tangent < 0.999999
            else 1e-6
        )


class _FinHeight(om.ExplicitComponent):
    """
    Computing the fin height from the hyperbolic tangent of the fin height, used for the regularization of the fin height
    sizing equation.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="finned_heat_sink_id",
            default=None,
            desc="Identifier of the finned heat sink",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        self.add_input(
            "inverse_hyperbolic_tangent",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_parameter",
            units="m**-1",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
            units="m",
            val=0.1,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height"
        ] = (
            inputs["inverse_hyperbolic_tangent"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + finned_heat_sink_id
                + ":fin_parameter"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
            "inverse_hyperbolic_tangent",
        ] = (
            1.0
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + finned_heat_sink_id
                + ":fin_parameter"
            ]
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_parameter",
        ] = (
            -inputs["inverse_hyperbolic_tangent"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + finned_heat_sink_id
                + ":fin_parameter"
            ]
            ** 2.0
        )
