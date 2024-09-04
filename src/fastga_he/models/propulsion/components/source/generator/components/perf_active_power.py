# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesActivePower(om.ExplicitComponent):
    """Computation of the electric active power required created by the generator."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        generator_id = self.options["generator_id"]

        self.add_input(
            "settings:propulsion:he_power_train:generator:" + generator_id + ":power_factor",
            val=1.0,
        )
        self.add_input(
            "apparent_power",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )

        self.add_output(
            "active_power",
            units="W",
            val=np.full(number_of_points, 500.0e3),
            shape=number_of_points,
        )

        self.declare_partials(
            of="active_power",
            wrt="settings:propulsion:he_power_train:generator:" + generator_id + ":power_factor",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="active_power",
            wrt="apparent_power",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]

        outputs["active_power"] = (
            inputs["apparent_power"]
            * inputs[
                "settings:propulsion:he_power_train:generator:" + generator_id + ":power_factor"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        generator_id = self.options["generator_id"]
        number_of_points = self.options["number_of_points"]

        partials["active_power", "apparent_power"] = (
            np.ones(number_of_points)
            * inputs[
                "settings:propulsion:he_power_train:generator:" + generator_id + ":power_factor"
            ]
        )
        partials[
            "active_power",
            "settings:propulsion:he_power_train:generator:" + generator_id + ":power_factor",
        ] = inputs["apparent_power"]
