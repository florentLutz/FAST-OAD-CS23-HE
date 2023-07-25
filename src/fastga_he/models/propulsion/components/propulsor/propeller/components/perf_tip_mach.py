# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials


class PerformancesTipMach(om.ExplicitComponent):
    """Computation of the squared mach number at the tip of the propeller."""

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("altitude", units="m", val=np.nan, shape=number_of_points)

        self.add_output(
            "tip_mach",
            val=0.4,
            shape=number_of_points,
            desc="Squared mach  number at the tip of the blades",
        )

        self.declare_partials(of="tip_mach", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        true_airspeed = inputs["true_airspeed"]

        sos = AtmosphereWithPartials(inputs["altitude"], altitude_in_feet=False).speed_of_sound
        omega = inputs["rpm"] * 2.0 * np.pi / 60.0

        tip_airspeed = true_airspeed ** 2.0 + (omega * diameter / 2.0) ** 2.0

        outputs["tip_mach"] = tip_airspeed / sos ** 2.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        true_airspeed = inputs["true_airspeed"]

        atm = AtmosphereWithPartials(inputs["altitude"], altitude_in_feet=False)
        sos = atm.speed_of_sound
        d_sos_d_altitude = atm.partial_speed_of_sound_altitude
        omega = inputs["rpm"] * 2.0 * np.pi / 60.0

        partials["tip_mach", "true_airspeed"] = np.diag(2.0 * true_airspeed / sos ** 2.0)
        partials["tip_mach", "rpm"] = np.diag(
            2.0 * omega * (diameter / 2.0) ** 2.0 / sos ** 2.0 * 2.0 * np.pi / 60.0
        )
        partials[
            "tip_mach", "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"
        ] = (2.0 * diameter * (omega / 2.0) ** 2.0 / sos ** 2.0)

        tip_airspeed = true_airspeed ** 2.0 + (omega * diameter / 2.0) ** 2.0
        partials["tip_mach", "altitude"] = np.diag(
            -2.0 * tip_airspeed / sos ** 3.0 * d_sos_d_altitude
        )
