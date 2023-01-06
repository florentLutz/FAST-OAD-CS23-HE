# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import Atmosphere


class PerformancesShaftPower(om.ExplicitComponent):
    """Computation of the power required on the shaft of the propeller."""

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

        self.add_input("power_coefficient", shape=number_of_points, val=np.nan)
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("altitude", units="m", val=0.0, shape=number_of_points)

        self.add_output("shaft_power_in", val=5e3, shape=number_of_points, units="W")

        self.declare_partials(
            of="shaft_power_in",
            wrt=[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
                "rpm",
                "power_coefficient",
            ],
            method="exact",
        )
        self.declare_partials(
            of="shaft_power_in",
            wrt="altitude",
            method="fd",
            form="central",
            step=1.0e2,
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        power_coefficient = inputs["power_coefficient"]

        rho = Atmosphere(inputs["altitude"], altitude_in_feet=False).density
        rps = inputs["rpm"] / 60.0

        outputs["shaft_power_in"] = power_coefficient * (rho * rps ** 3.0 * diameter ** 5.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        power_coefficient = inputs["power_coefficient"]

        rho = Atmosphere(inputs["altitude"], altitude_in_feet=False).density
        rps = inputs["rpm"] / 60.0

        partials["shaft_power_in", "power_coefficient"] = np.diag(
            rho * rps ** 3.0 * diameter ** 5.0
        )
        partials[
            "shaft_power_in",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = (
            5.0 * power_coefficient * rho * rps ** 3.0 * diameter ** 4.0
        )
        partials["shaft_power_in", "rpm"] = (
            np.diag(3.0 * power_coefficient * rho * rps ** 2.0 * diameter ** 5.0) / 60.0
        )
