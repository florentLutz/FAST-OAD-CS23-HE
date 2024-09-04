# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


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
        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)

        self.add_input(
            "convergence:propulsion:he_power_train:propeller:" + propeller_id + ":min_power",
            units="W",
            val=5.0e3,
            desc="Convergence parameter used to aid convergence since, if power is too low in the "
            "network, the code will have trouble converging",
        )

        self.add_output("shaft_power_in", val=50e3, shape=number_of_points, units="W")

        self.declare_partials(
            of="shaft_power_in",
            wrt=[
                "convergence:propulsion:he_power_train:propeller:" + propeller_id + ":min_power",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="shaft_power_in",
            wrt=["rpm", "density", "power_coefficient"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        power_coefficient = inputs["power_coefficient"]
        rho = inputs["density"]
        rps = inputs["rpm"] / 60.0

        min_shaft_power = inputs[
            "convergence:propulsion:he_power_train:propeller:" + propeller_id + ":min_power"
        ]

        shaft_power = np.clip(
            power_coefficient * (rho * rps**3.0 * diameter**5.0), min_shaft_power, None
        )

        outputs["shaft_power_in"] = shaft_power

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        power_coefficient = inputs["power_coefficient"]

        rho = inputs["density"]
        rps = inputs["rpm"] / 60.0

        min_shaft_power = inputs[
            "convergence:propulsion:he_power_train:propeller:" + propeller_id + ":min_power"
        ]

        shaft_power = power_coefficient * (rho * rps**3.0 * diameter**5.0)

        partials["shaft_power_in", "power_coefficient"] = np.where(
            shaft_power < min_shaft_power, 1e-6, rho * rps**3.0 * diameter**5.0
        )
        partials[
            "shaft_power_in",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = np.where(
            shaft_power < min_shaft_power,
            1e-6,
            5.0 * power_coefficient * rho * rps**3.0 * diameter**4.0,
        )
        partials["shaft_power_in", "rpm"] = np.where(
            shaft_power < min_shaft_power,
            1e-6,
            3.0 * power_coefficient * rho * rps**2.0 * diameter**5.0 / 60.0,
        )
        partials["shaft_power_in", "density"] = np.where(
            shaft_power < min_shaft_power,
            1e-6,
            power_coefficient * rps**3.0 * diameter**5.0,
        )
        partials[
            "shaft_power_in",
            "convergence:propulsion:he_power_train:propeller:" + propeller_id + ":min_power",
        ] = np.where(shaft_power < min_shaft_power, 1.0, 1e-6)
