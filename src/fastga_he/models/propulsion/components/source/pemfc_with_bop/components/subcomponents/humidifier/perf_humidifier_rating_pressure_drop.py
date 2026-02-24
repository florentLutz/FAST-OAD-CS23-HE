# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..fluid_characteristics import FluidDensity


class PerformancesHumidifierRatingPressureDrop(om.Group):
    """
    Maximum pressure drop computation of the humidifier during mission.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_subsystem(
            "compressed_air_density",
            FluidDensity(),
            promotes=[
                (
                    "fluid_pressure",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":operating_pressure",
                ),
                (
                    "fluid_temperature",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":oxidizer_temperature",
                ),
            ],
        )
        self.add_subsystem(
            "humidifier_rating_pressure_drop",
            _HumidifierRatingPressureDrop(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )

        self.connect(
            "compressed_air_density.fluid_density",
            "humidifier_rating_pressure_drop.compressed_air_density",
        )


class _HumidifierRatingPressureDrop(om.ExplicitComponent):
    """
    Computes the maximum pressure drop of the humidifier during mission.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            "compressed_air_density",
            val=np.nan,
            units="kg/m**3",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
            val=np.nan,
            units="kg/s",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":humidifier:max_pressure_drop",
            val=1e4,
            units="Pa",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressed_air_density = inputs["compressed_air_density"]
        air_consumption_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]
        max_volumetric_flow_rate = air_consumption_max / compressed_air_density

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":humidifier:max_pressure_drop"
        ] = (
            (-3.628 * 1e5 * air_consumption_max**2.0 + 1.995 * 1e5 * air_consumption_max - 4000.0)
            if max_volumetric_flow_rate <= 0.06815
            else max_volumetric_flow_rate * 12000.0 / 0.083
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        compressed_air_density = inputs["compressed_air_density"]
        air_consumption_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]
        max_volumetric_flow_rate = air_consumption_max / compressed_air_density

        if max_volumetric_flow_rate <= 0.06815:
            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":humidifier:max_pressure_drop",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_consumption_max",
            ] = -7.256 * 1e5 * air_consumption_max + 1.995 * 1e5

            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":humidifier:max_pressure_drop",
                "compressed_air_density",
            ] = 0.0

        else:
            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":humidifier:max_pressure_drop",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":air_consumption_max",
            ] = 12000.0 / 0.083 / compressed_air_density

            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":humidifier:max_pressure_drop",
                "compressed_air_density",
            ] = -12000.0 / 0.083 * air_consumption_max / compressed_air_density**2.0
