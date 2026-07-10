# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPressureEfficiencyDifference(om.ExplicitComponent):
    """
    Computation of the flush_inlet pressure efficiency difference due to flow condition.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "pressure_efficiency_difference_factor",
            val=np.nan,
            units="unitless",
            shape=number_of_points,
        )

        self.add_output(
            "pressure_efficiency_difference",
            val=-0.05,
            units="unitless",
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
        pressure_efficiency_difference_factor = inputs["pressure_efficiency_difference_factor"]

        conditions = [
            pressure_efficiency_difference_factor < 0.0,
            pressure_efficiency_difference_factor > 0.0,
        ]

        delta_pressure_efficiency = [
            0.2596 * pressure_efficiency_difference_factor + 0.0086,
            (
                -0.3174 * pressure_efficiency_difference_factor**3.0
                + 0.5455 * pressure_efficiency_difference_factor**2.0
                - 0.3274 * pressure_efficiency_difference_factor
                + 0.0144
            ),
        ]

        outputs["pressure_efficiency_difference"] = np.select(
            conditions, delta_pressure_efficiency, default=0.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pressure_efficiency_difference_factor = inputs["pressure_efficiency_difference_factor"]

        conditions = [
            pressure_efficiency_difference_factor < 0.0,
            pressure_efficiency_difference_factor > 0.0,
        ]

        ddelta_eta = [
            0.2596,
            (
                -0.9522 * pressure_efficiency_difference_factor**2.0
                + 1.0910 * pressure_efficiency_difference_factor
                - 0.3274
            ),
        ]

        partials["pressure_efficiency_difference", "pressure_efficiency_difference_factor"] = (
            np.select(conditions, ddelta_eta, default=0.0)
        )
