# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHeatSinkFinLength(om.ImplicitComponent):
    """
    Computing heat sink fin length based on PEMFC length.
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
            val=np.nan,
            desc="Thermal conductivity of the fin material",
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

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
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
        fin_height = outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height"
        ]

        residuals[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height"
        ] = (
            number_of_fins
            * fin_heat_transfer_parameter
            * (
                np.sinh(fin_parameter * fin_height)
                + (
                    convection_heat_transfer_coefficient
                    / (fin_parameter * conduction_coefficient)
                    * np.cosh(fin_parameter * fin_height)
                )
            )
            / (
                np.cosh(fin_parameter * fin_height)
                + (
                    convection_heat_transfer_coefficient
                    / (fin_parameter * conduction_coefficient)
                    * np.sinh(fin_parameter * fin_height)
                )
            )
            - design_dissipation_power
        )

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

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
        fin_height = outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height"
        ]

        jacobian[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":number_of_fins",
        ] = (
            fin_heat_transfer_parameter
            * (
                np.sinh(fin_parameter * fin_height)
                + (
                    convection_heat_transfer_coefficient
                    / (fin_parameter * conduction_coefficient)
                    * np.cosh(fin_parameter * fin_height)
                )
            )
            / (
                np.cosh(fin_parameter * fin_height)
                + (
                    convection_heat_transfer_coefficient
                    / (fin_parameter * conduction_coefficient)
                    * np.sinh(fin_parameter * fin_height)
                )
            )
        )
        jacobian[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_heat_transfer_parameter",
        ] = (
            number_of_fins
            * (
                np.sinh(fin_parameter * fin_height)
                + (
                    convection_heat_transfer_coefficient
                    / (fin_parameter * conduction_coefficient)
                    * np.cosh(fin_parameter * fin_height)
                )
            )
            / (
                np.cosh(fin_parameter * fin_height)
                + (
                    convection_heat_transfer_coefficient
                    / (fin_parameter * conduction_coefficient)
                    * np.sinh(fin_parameter * fin_height)
                )
            )
        )

        jacobian[
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
            -number_of_fins
            * fin_heat_transfer_parameter
            * (
                (
                    fin_height * conduction_coefficient**2.0 * fin_parameter**2.0
                    - convection_heat_transfer_coefficient * conduction_coefficient
                    - fin_height * convection_heat_transfer_coefficient**2.0
                )
                * (
                    np.sinh(fin_height * fin_parameter) ** 2.0
                    - np.cosh(fin_height * fin_parameter) ** 2.0
                )
            )
            / (
                convection_heat_transfer_coefficient * np.sinh(fin_height * fin_parameter)
                + conduction_coefficient * fin_parameter * np.cosh(fin_height * fin_parameter)
            )
            ** 2.0
        )

        jacobian[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":design_dissipation_power",
        ] = -1.0

        jacobian[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":convection_heat_transfer_coefficient",
        ] = (
            -number_of_fins
            * fin_heat_transfer_parameter
            * (
                conduction_coefficient
                * fin_parameter
                * (
                    np.sinh(fin_parameter * fin_height) ** 2.0
                    - np.cosh(fin_parameter * fin_height) ** 2.0
                )
            )
            / (
                np.sinh(fin_parameter * fin_height) * convection_heat_transfer_coefficient
                + conduction_coefficient * fin_parameter * np.cosh(fin_parameter * fin_height)
            )
            ** 2.0
        )

        jacobian[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":conduction_coefficient",
        ] = (
            number_of_fins
            * fin_heat_transfer_parameter
            * (
                convection_heat_transfer_coefficient
                * fin_parameter
                * (
                    np.sinh(fin_parameter * fin_height) ** 2.0
                    - np.cosh(fin_parameter * fin_height) ** 2.0
                )
            )
            / (
                np.sinh(fin_parameter * fin_height) * convection_heat_transfer_coefficient
                + conduction_coefficient * fin_parameter * np.cosh(fin_parameter * fin_height)
            )
            ** 2.0
        )

        jacobian[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_height",
        ] = (
            -number_of_fins
            * fin_heat_transfer_parameter
            * (
                fin_parameter
                * (
                    conduction_coefficient**2.0 * fin_parameter**2.0
                    - convection_heat_transfer_coefficient**2.0
                )
                * (
                    np.sinh(fin_parameter * fin_height) ** 2.0
                    - np.cosh(fin_parameter * fin_height) ** 2.0
                )
            )
            / (
                np.sinh(fin_parameter * fin_height) * convection_heat_transfer_coefficient
                + conduction_coefficient * fin_parameter * np.cosh(fin_parameter * fin_height)
            )
            ** 2.0
        )
