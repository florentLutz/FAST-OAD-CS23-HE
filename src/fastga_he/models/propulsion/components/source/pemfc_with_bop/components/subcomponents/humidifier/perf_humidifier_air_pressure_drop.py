# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesHumidifierRatingPressureDrop(om.ExplicitComponent):
    """
    Computes the maximum pressure drop of the humidifier during mission.
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
            name="humidifier_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        humidifier_id = self.options["humidifier_id"]

        self.add_input(
            "humidifier_air_density", val=np.nan, units="kg/m**3", shape=number_of_points
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
            + humidifier_id
            + ":air_pressure_drop",
            val=1e4,
            units="Pa",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        humidifier_id = self.options["humidifier_id"]

        humidifier_air_density = inputs["humidifier_air_density"]
        air_consumption = inputs["air_consumption"]
        volumetric_flow_rate = air_consumption / humidifier_air_density

        conditions = [
            (volumetric_flow_rate > 0.054) & (volumetric_flow_rate <= 0.06815),
            volumetric_flow_rate > 0.06815,
        ]

        choices = [
            -3.628 * 1e5 * air_consumption**2.0 + 1.995 * 1e5 * air_consumption - 4000.0,
            volumetric_flow_rate * 12000.0 / 0.083,
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + humidifier_id
            + ":air_pressure_drop"
        ] = np.select(conditions, choices, default=0.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        humidifier_id = self.options["humidifier_id"]

        humidifier_air_density = inputs["humidifier_air_density"]
        air_consumption = inputs["air_consumption"]
        volumetric_flow_rate = air_consumption / humidifier_air_density

        conditions = [
            (volumetric_flow_rate > 0.054) & (volumetric_flow_rate <= 0.06815),
            volumetric_flow_rate > 0.06815,
        ]

        drho = [
            np.zeros_like(air_consumption),
            -air_consumption / humidifier_air_density**2.0 * 12000.0 / 0.083,
        ]

        dair_consumption = [
            -7.256 * 1e5 * air_consumption + 1.995 * 1e5,
            12000.0 / 0.083 / humidifier_air_density,
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + humidifier_id
            + ":air_pressure_drop",
            "humidifier_air_density",
        ] = np.select(conditions, drho, default=0.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + humidifier_id
            + ":air_pressure_drop",
            "air_consumption",
        ] = np.select(conditions, dair_consumption, default=0.0)
