# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesTemperatureDerivative(om.ExplicitComponent):
    """Computation of the derivative of the temperature inside the casing. Assumes lumped casing
    and that all losses are dissipated as heat"""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "inverter_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            "heat_sink_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the heat sink",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":casing:thermal_resistance",
            units="K/W",
            val=np.nan,
            desc="Thermal resistance between the casing and the heat sink",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:heat_capacity",
            units="J/degK",
            val=np.nan,
            desc="Heat capacity of one casing",
        )
        self.add_input(
            "losses_inverter",
            units="W",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            "inverter_temperature_time_derivative",
            val=np.full(number_of_points, 0.0),
            units="degK/s",
            desc="temperature inside of the inverter",
            shape=number_of_points,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        temp_inv = inputs["inverter_temperature"]
        temp_hs = inputs["heat_sink_temperature"]
        r_th_js = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:thermal_resistance"
        ]
        hc_casing = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:heat_capacity"
        ]

        # We assume that module dissipate their heat the same way, so we will only look at the
        # evolution in one module
        losses_one_module = inputs["losses_inverter"] / 3.0

        q_cool = (temp_inv - temp_hs) / r_th_js

        d_temp_d_t = (losses_one_module - q_cool) / hc_casing

        outputs["inverter_temperature_time_derivative"] = d_temp_d_t

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        temp_inv = inputs["inverter_temperature"]
        temp_hs = inputs["heat_sink_temperature"]
        r_th_js = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:thermal_resistance"
        ]
        hc_casing = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:heat_capacity"
        ]

        # We assume that module dissipate their heat the same way, so we will only look at the
        # evolution in one module
        losses_one_module = inputs["losses_inverter"] / 3.0

        q_cool = (temp_inv - temp_hs) / r_th_js

        partials["inverter_temperature_time_derivative", "inverter_temperature"] = -1.0 / (
            hc_casing * r_th_js
        )
        partials["inverter_temperature_time_derivative", "heat_sink_temperature"] = 1.0 / (
            hc_casing * r_th_js
        )
        partials[
            "inverter_temperature_time_derivative",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:thermal_resistance",
        ] = (temp_inv - temp_hs) / (hc_casing * r_th_js ** 2.0)
        partials[
            "inverter_temperature_time_derivative",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:heat_capacity",
        ] = (
            -(losses_one_module - q_cool) / hc_casing ** 2.0
        )
        partials[
            "inverter_temperature_time_derivative",
            "losses_inverter",
        ] = 1.0 / (hc_casing * 3.0)
