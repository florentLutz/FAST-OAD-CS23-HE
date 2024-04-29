# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamTurboshaftDeltaCd(om.ExplicitComponent):
    """The exhaust of the thrust will be accounted for through a negative drag coefficient."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("exhaust_thrust", units="N", val=np.nan, shape=number_of_points)

        self.add_output("delta_Cd", val=1e-5, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        wing_area = inputs["data:geometry:wing:area"]

        exhaust_thrust = inputs["exhaust_thrust"]

        delta_cd = -exhaust_thrust / (0.5 * density * true_airspeed ** 2.0 * wing_area)

        outputs["delta_Cd"] = delta_cd

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        wing_area = inputs["data:geometry:wing:area"]

        exhaust_thrust = inputs["exhaust_thrust"]

        partials["delta_Cd", "density"] = np.diag(
            exhaust_thrust / (0.5 * density ** 2.0 * true_airspeed ** 2.0 * wing_area)
        )
        partials["delta_Cd", "true_airspeed"] = 2.0 * np.diag(
            exhaust_thrust / (0.5 * density * true_airspeed ** 3.0 * wing_area)
        )
        partials["delta_Cd", "data:geometry:wing:area"] = exhaust_thrust / (
            0.5 * density * true_airspeed ** 2.0 * wing_area ** 2.0
        )
        partials["delta_Cd", "exhaust_thrust"] = -np.diag(
            1.0 / (0.5 * density * true_airspeed ** 2.0 * wing_area)
        )
