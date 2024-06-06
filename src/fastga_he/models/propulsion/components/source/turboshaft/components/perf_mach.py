# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials


class PerformancesMach(om.ExplicitComponent):
    """Computation of free-stream Mach number."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("altitude", units="m", val=np.nan, shape=number_of_points)

        self.add_output("mach", val=0.2, shape=number_of_points, lower=0.0)

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        true_airspeed = inputs["true_airspeed"]
        altitude = inputs["altitude"]

        speed_of_sound = AtmosphereWithPartials(altitude, altitude_in_feet=False).speed_of_sound

        outputs["mach"] = true_airspeed / speed_of_sound

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        true_airspeed = inputs["true_airspeed"]
        altitude = inputs["altitude"]

        atm = AtmosphereWithPartials(altitude, altitude_in_feet=False)

        speed_of_sound = atm.speed_of_sound
        d_sos_d_altitude = atm.partial_speed_of_sound_altitude

        partials["mach", "true_airspeed"] = 1.0 / speed_of_sound
        partials["mach", "altitude"] = -true_airspeed / speed_of_sound ** 2.0 * d_sos_d_altitude
