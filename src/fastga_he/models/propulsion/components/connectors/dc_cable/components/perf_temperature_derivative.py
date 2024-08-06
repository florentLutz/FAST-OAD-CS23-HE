# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTemperatureDerivative(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):
        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "conduction_losses",
            units="W",
            desc="Joule losses in one cable of the harness",
            val=np.full(number_of_points, np.nan),
        )
        self.add_input(
            "cable_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            "heat_transfer_coefficient",
            val=np.full(number_of_points, 11.0),
            units="W/m**2/degK",
            desc="Heat transfer coefficient between cable and outside medium",
            shape=number_of_points,
        )
        self.add_input(
            "exterior_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature outside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity",
            val=np.nan,
            units="J/degK",
        )

        self.add_output(
            "cable_temperature_time_derivative",
            val=np.full(number_of_points, 0.0),
            units="degK/s",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )

        self.declare_partials(
            of="cable_temperature_time_derivative",
            wrt=[
                "exterior_temperature",
                "heat_transfer_coefficient",
                "cable_temperature",
                "conduction_losses",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="cable_temperature_time_derivative",
            wrt=[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:heat_capacity",
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        cable_radius = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius"
        ]
        cable_length = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"
        ]
        cable_heat_capacity = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:heat_capacity"
        ]

        temp_cable = inputs["cable_temperature"]
        h = inputs["heat_transfer_coefficient"]

        temp_ext = inputs["exterior_temperature"]

        q_c = inputs["conduction_losses"]
        q_inf = 2.0 * cable_radius * cable_length * np.pi * h * (temp_cable - temp_ext)

        d_temp_d_t = (q_c - q_inf) / cable_heat_capacity

        outputs["cable_temperature_time_derivative"] = d_temp_d_t

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        cable_radius = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius"
        ]
        cable_length = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"
        ]
        cable_heat_capacity = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:heat_capacity"
        ]

        temp_cable = inputs["cable_temperature"]
        h = inputs["heat_transfer_coefficient"]

        temp_ext = inputs["exterior_temperature"]

        q_c = inputs["conduction_losses"]
        q_inf = 2.0 * cable_radius * cable_length * np.pi * h * (temp_cable - temp_ext)

        partials["cable_temperature_time_derivative", "conduction_losses"] = (
            np.ones(number_of_points) / cable_heat_capacity
        )

        partials["cable_temperature_time_derivative", "cable_temperature"] = -(
            2.0 * cable_radius * cable_length * np.pi * h / cable_heat_capacity
        )
        partials["cable_temperature_time_derivative", "heat_transfer_coefficient"] = -(
            2.0
            * cable_radius
            * cable_length
            * np.pi
            * (temp_cable - temp_ext)
            / cable_heat_capacity
        )
        partials[
            "cable_temperature_time_derivative",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
        ] = -(2.0 * cable_length * h * np.pi * (temp_cable - temp_ext) / cable_heat_capacity)
        partials[
            "cable_temperature_time_derivative",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
        ] = -(2.0 * cable_radius * h * np.pi * (temp_cable - temp_ext) / cable_heat_capacity)
        partials["cable_temperature_time_derivative", "exterior_temperature"] = (
            2.0 * cable_radius * cable_length * np.pi * h / cable_heat_capacity
        )

        partials[
            "cable_temperature_time_derivative",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity",
        ] = -(q_c - q_inf) / cable_heat_capacity**2.0
