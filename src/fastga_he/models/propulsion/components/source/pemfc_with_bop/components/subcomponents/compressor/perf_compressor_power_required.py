# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCompressorPowerRequired(om.ExplicitComponent):
    """
    Computation of the amount of power required for compressor.
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

        self.add_input(
            "compressor_pressure_ratio",
            val=np.nan,
            units="unitless",
            shape=number_of_points,
        )
        self.add_input(
            "exterior_temperature",
            val=np.nan,
            units="K",
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
        self.add_input(
            "compressed_air_specific_heat_capacity",
            val=np.nan,
            units="J/kg/K",
            shape=number_of_points,
        )
        self.add_input(
            "air_consumption",
            val=np.nan,
            units="kg/s",
            shape=number_of_points,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
            val=0.0,
            units="W",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        self.declare_partials(
            of="*",
            wrt=[
                "compressor_pressure_ratio",
                "exterior_temperature",
                "compressed_air_specific_heat_capacity",
                "air_consumption",
            ],
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
        compressor_pressure_ratio = inputs["compressor_pressure_ratio"]
        specific_heat_ratio = inputs[
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
        compressed_air_specific_heat_capacity = inputs["compressed_air_specific_heat_capacity"]
        air_consumption = inputs["air_consumption"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required"
        ] = (
            compressed_air_specific_heat_capacity
            * exterior_temperature
            / efficiency
            * (
                compressor_pressure_ratio ** ((specific_heat_ratio - 1.0) / specific_heat_ratio)
                - 1.0
            )
            * air_consumption
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        exterior_temperature = inputs["exterior_temperature"]
        compressor_pressure_ratio = inputs["compressor_pressure_ratio"]
        specific_heat_ratio = inputs[
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
        compressed_air_specific_heat_capacity = inputs["compressed_air_specific_heat_capacity"]
        air_consumption = inputs["air_consumption"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
            "exterior_temperature",
        ] = (
            compressed_air_specific_heat_capacity
            / efficiency
            * (
                compressor_pressure_ratio ** ((specific_heat_ratio - 1.0) / specific_heat_ratio)
                - 1.0
            )
            * air_consumption
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
            "compressed_air_specific_heat_capacity",
        ] = (
            exterior_temperature
            / efficiency
            * (
                compressor_pressure_ratio ** ((specific_heat_ratio - 1.0) / specific_heat_ratio)
                - 1.0
            )
            * air_consumption
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
            "air_consumption",
        ] = (
            compressed_air_specific_heat_capacity
            * exterior_temperature
            / efficiency
            * (
                compressor_pressure_ratio ** ((specific_heat_ratio - 1.0) / specific_heat_ratio)
                - 1.0
            )
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":efficiency",
        ] = -(
            compressed_air_specific_heat_capacity
            * exterior_temperature
            / efficiency**2.0
            * (
                compressor_pressure_ratio ** ((specific_heat_ratio - 1.0) / specific_heat_ratio)
                - 1.0
            )
            * air_consumption
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":specific_heat_ratio",
        ] = (
            compressed_air_specific_heat_capacity
            * exterior_temperature
            / efficiency
            * (
                compressor_pressure_ratio ** ((specific_heat_ratio - 1.0) / specific_heat_ratio)
                * np.log(compressor_pressure_ratio)
                / specific_heat_ratio**2.0
            )
            * air_consumption
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
            "compressor_pressure_ratio",
        ] = (
            compressed_air_specific_heat_capacity
            * exterior_temperature
            * air_consumption
            / efficiency
            * ((specific_heat_ratio - 1.0) / specific_heat_ratio)
            * compressor_pressure_ratio ** ((specific_heat_ratio - 1.0) / specific_heat_ratio - 1.0)
        )
