# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCompressorPressureRatio(om.ExplicitComponent):
    """
    Computation of the pressure ratio of the compressor.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("ambient_pressure", units="Pa", val=np.full(number_of_points, np.nan))
        self.add_input(
            "compressor_pressure_supply",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )

        self.add_output(
            "compressor_pressure_ratio",
            val=1.2,
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
        outputs["compressor_pressure_ratio"] = (
            inputs["compressor_pressure_supply"] / inputs["ambient_pressure"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        partials["compressor_pressure_ratio", "compressor_pressure_supply"] = (
            1.0 / inputs["ambient_pressure"]
        )
        partials["compressor_pressure_ratio", "ambient_pressure"] = (
            -inputs["compressor_pressure_supply"] / inputs["ambient_pressure"] ** 2.0
        )
