# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesThroatPressure(om.ExplicitComponent):
    """
    Computation of the total throat pressure.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("ambient_pressure", units="Pa", val=np.full(number_of_points, np.nan))
        self.add_input(
            "ambient_total_pressure",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )
        self.add_input(
            "inlet_efficiency",
            val=np.nan,
            units="unitless",
            shape=number_of_points,
        )

        self.add_output(
            "throat_total_pressure",
            val=0.3,
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
        static_pressure = inputs["ambient_pressure"]
        ambient_total_pressure = inputs["ambient_total_pressure"]
        inlet_efficiency = inputs["inlet_efficiency"]

        outputs["throat_total_pressure"] = (
            inlet_efficiency * (ambient_total_pressure - static_pressure) + static_pressure
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        static_pressure = inputs["ambient_pressure"]
        ambient_total_pressure = inputs["ambient_total_pressure"]
        inlet_efficiency = inputs["inlet_efficiency"]

        partials["throat_total_pressure", "ambient_pressure"] = 1.0 - inlet_efficiency

        partials["throat_total_pressure", "ambient_total_pressure"] = inlet_efficiency

        partials["throat_total_pressure", "inlet_efficiency"] = (
            ambient_total_pressure - static_pressure
        )
