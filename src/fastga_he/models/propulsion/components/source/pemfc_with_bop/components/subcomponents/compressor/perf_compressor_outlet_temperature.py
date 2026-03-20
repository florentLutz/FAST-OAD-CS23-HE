# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCompressorOutletTemperature(om.ExplicitComponent):
    """
    Computation of the outlet temperature of the compressor. This is calculated with the
    isentropic flow assumption.
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

        self.add_input("exterior_temperature", units="K", val=np.full(number_of_points, np.nan))
        self.add_input(
            "compressor_pressure_ratio",
            val=np.nan,
            units="unitless",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio",
            val=1.4,
            units="unitless",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":efficiency",
            val=0.85,
            units="unitless",
        )

        self.add_output(
            "compressor_outlet_temperature",
            val=300.0,
            units="K",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        self.declare_partials(
            of="*",
            wrt=["compressor_pressure_ratio", "exterior_temperature"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + compressor_id
                + ":specific_heat_ratio",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + compressor_id
                + ":efficiency",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        exterior_temperature = inputs["exterior_temperature"]
        gamma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio"
        ]
        efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":efficiency"
        ]
        pressure_ratio = inputs["compressor_pressure_ratio"]

        outputs["compressor_outlet_temperature"] = exterior_temperature * (
            1.0 + (pressure_ratio ** ((gamma - 1.0) / gamma) - 1.0) / efficiency
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        exterior_temperature = inputs["exterior_temperature"]
        gamma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio"
        ]
        efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":efficiency"
        ]
        pressure_ratio = inputs["compressor_pressure_ratio"]

        partials["compressor_outlet_temperature", "exterior_temperature"] = (
            1.0 + (pressure_ratio ** ((gamma - 1.0) / gamma) - 1.0) / efficiency
        )

        partials["compressor_outlet_temperature", "compressor_pressure_ratio"] = (
            2.0
            * exterior_temperature
            * (pressure_ratio ** ((gamma - 1.0) / gamma - 1.0) / efficiency)
            * (gamma - 1.0)
            / (gamma * pressure_ratio)
        )

        partials[
            "compressor_outlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio",
        ] = exterior_temperature * (
            pressure_ratio ** ((gamma - 1.0) / gamma)
            * np.log(pressure_ratio)
            / (efficiency * gamma**2.0)
        )

        partials[
            "compressor_outlet_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":efficiency",
        ] = (
            -exterior_temperature
            * (pressure_ratio ** ((gamma - 1.0) / gamma) - 1.0)
            / efficiency**2.0
        )
