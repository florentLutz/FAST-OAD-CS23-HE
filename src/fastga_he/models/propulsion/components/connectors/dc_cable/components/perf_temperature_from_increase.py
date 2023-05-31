# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTemperatureFromIncrease(om.ExplicitComponent):
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
            "cable_temperature_increase",
            val=np.full(number_of_points, np.nan),
            units="degK",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:initial_temperature",
            units="degK",
            val=np.nan,
        )

        self.add_output(
            "cable_temperature",
            val=np.full(number_of_points, 288.15),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
            lower=1.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        d_temp = inputs["cable_temperature_increase"]
        initial_temperature = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:initial_temperature"
        ]

        temperature_profile = np.full(number_of_points, initial_temperature) + np.cumsum(
            np.concatenate((np.zeros(1), d_temp[:-1]))
        )

        outputs["cable_temperature"] = temperature_profile

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        partials[
            "cable_temperature",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:initial_temperature",
        ] = np.ones(number_of_points)
        partials["cable_temperature", "cable_temperature_increase"] = np.tri(
            number_of_points, number_of_points
        ) - np.eye(number_of_points)
