# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAirDynamicPressure(om.ExplicitComponent):
    """
    Computation of the dynamic pressure of the ambient air, which varies based on the change of
    operating altitude and the true air speed of the aircraft.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("density", units="kg/m**3", val=np.zeros(number_of_points))
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)

        self.add_output(
            name="dynamic_pressure",
            units="Pa",
            val=np.full(number_of_points, 7200.0),
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
        outputs["dynamic_pressure"] = 0.5 * inputs["density"] * inputs["true_airspeed"] ** 2.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["dynamic_pressure", "density"] = 0.5 * inputs["true_airspeed"] ** 2.0

        partials["dynamic_pressure", "true_airspeed"] = inputs["density"] * inputs["true_airspeed"]
