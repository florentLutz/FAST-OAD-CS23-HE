# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import Atmosphere


class PerformancesThrustCoefficient(om.ExplicitComponent):
    """Computation of the thrust coefficient of the propeller."""

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
        self.add_input("altitude", units="m", val=0.0, shape=number_of_points)
        self.add_input("thrust", units="N", val=1500, shape=number_of_points)

        self.add_output("thrust_coefficient", val=0.05, shape=number_of_points)

        self.declare_partials(
            of="thrust_coefficient",
            wrt=[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
                "rpm",
                "thrust",
            ],
            method="exact",
        )
        self.declare_partials(
            of="thrust_coefficient", wrt="altitude", method="fd", form="central", step=1.0e2
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]

        rho = Atmosphere(inputs["altitude"], altitude_in_feet=False).density
        rps = inputs["rpm"] / 60.0

        outputs["thrust_coefficient"] = inputs["thrust"] / (rho * rps ** 2.0 * diameter ** 4.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]

        rho = Atmosphere(inputs["altitude"], altitude_in_feet=False).density
        rps = inputs["rpm"] / 60.0

        partials["thrust_coefficient", "thrust"] = np.diag(
            1.0 / (rho * rps ** 2.0 * diameter ** 4.0)
        )
        partials["thrust_coefficient", "rpm"] = (
            -2.0 * np.diag(inputs["thrust"] / (rho * rps ** 3 * diameter ** 4.0)) / 60.0
        )
        partials[
            "thrust_coefficient",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = (
            -4.0 * inputs["thrust"] / (rho * rps ** 2.0 * diameter ** 5.0)
        )