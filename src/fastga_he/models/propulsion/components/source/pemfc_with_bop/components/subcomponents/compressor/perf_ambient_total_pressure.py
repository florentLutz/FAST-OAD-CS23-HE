# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAmbientTotalPressure(om.ExplicitComponent):
    """
    Computation of the ambient total pressure.
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
            name="compressor_id",
            default=None,
            desc="Identifier of the compressor",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        self.add_input("mach", val=np.nan, shape=number_of_points)
        self.add_input("ambient_pressure", units="Pa", val=np.full(number_of_points, np.nan))
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio",
            val=1.4,
            units="unitless",
        )

        self.add_output(
            "ambient_total_pressure",
            val=0.3,
            units="Pa",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        self.declare_partials(
            of="*",
            wrt=["mach", "ambient_pressure"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        static_pressure = inputs["ambient_pressure"]
        mach = inputs["mach"]
        gamma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio"
        ]

        outputs["ambient_total_pressure"] = static_pressure * (
            1.0 + 0.5 * (gamma - 1.0) * mach**2.0
        ) ** (gamma / (gamma - 1.0))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        static_pressure = inputs["ambient_pressure"]
        mach = inputs["mach"]
        gamma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio"
        ]

        partials["ambient_total_pressure", "ambient_pressure"] = (
            1.0 + 0.5 * (gamma - 1.0) * mach**2.0
        ) ** (gamma / (gamma - 1.0))

        partials["ambient_total_pressure", "mach"] = (
            static_pressure
            * gamma
            * mach
            * (((gamma - 1.0) * mach**2.0) * 0.5 + 1.0) ** (gamma / (gamma - 1.0) - 1.0)
        )

        partials[
            "ambient_total_pressure",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio",
        ] = (
            static_pressure
            * ((mach**2.0 * (gamma - 1.0)) * 0.5 + 1.0) ** (gamma / (gamma - 1.0))
            * (
                np.log((mach**2.0 * (gamma - 1.0)) * 0.5 + 1.0)
                * (1.0 / (gamma - 1.0) - gamma / (gamma - 1.0) ** 2.0)
                + (mach**2.0 * gamma)
                / (2.0 * ((mach**2.0 * (gamma - 1.0)) / 2.0 + 1.0) * (gamma - 1.0))
            )
        )
