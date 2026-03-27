# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesHeatExchangerNTU(om.ImplicitComponent):
    """
    Computation of the number of transfer unit of the heat exchanger.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input("heat_capacity_ratio", units="unitless", val=np.nan)
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":effectiveness",
            units="unitless",
            val=0.98,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":NTU",
            units="unitless",
            val=4.2,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        c_ratio = float(inputs["heat_capacity_ratio"])
        eps_hex = float(
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":effectiveness"
            ]
        )
        ntu = outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":NTU"
        ]

        residuals[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":NTU"
        ] = (
            1.0
            - np.exp((1.0 / c_ratio) * (ntu**0.22) * (np.exp(-c_ratio * ntu**0.78) - 1.0))
            - eps_hex
        )

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        c_ratio = inputs["heat_capacity_ratio"]
        ntu = outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":NTU"
        ]

        # Intermediate quantities for clarity
        exp_inner = np.exp(-c_ratio * ntu**0.78)  # exp(-Cr * N^0.78)
        A = (1.0 / c_ratio) * (ntu**0.22) * (exp_inner - 1.0)
        exp_A = np.exp(A)  # exp(A)

        # dR/d_eps = -1
        jacobian[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":NTU",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":effectiveness",
        ] = -1.0

        dA_dCr = (
            -(1.0 / c_ratio**2) * (ntu**0.22) * (exp_inner - 1.0)
            - (1.0 / c_ratio) * ntu * exp_inner
        )
        jacobian[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":NTU",
            "heat_capacity_ratio",
        ] = -exp_A * dA_dCr

        dA_dN = (0.22 / c_ratio) * (ntu**-0.78) * (exp_inner - 1.0) - 0.78 * exp_inner
        jacobian[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":NTU",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":NTU",
        ] = -exp_A * dA_dN
