# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials


class PerformancesBladeReynoldsNumber(om.ExplicitComponent):
    """Computation of the reynolds number corresponding to the diameter of the propeller."""

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
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("altitude", units="m", val=0.0, shape=number_of_points)

        self.add_output("reynolds_D", val=2e7, shape=number_of_points)

        self.declare_partials(
            of="reynolds_D",
            wrt="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="reynolds_D",
            wrt=["true_airspeed", "rpm", "altitude"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        true_airspeed = inputs["true_airspeed"]

        viscosity = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).kinematic_viscosity
        omega = inputs["rpm"] * 2.0 * np.pi / 60.0

        outputs["reynolds_D"] = (
            np.sqrt(true_airspeed**2.0 + (omega * diameter / 2.0) ** 2.0) * diameter / viscosity
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        true_airspeed = inputs["true_airspeed"]

        atm = AtmosphereWithPartials(inputs["altitude"], altitude_in_feet=False)
        viscosity = atm.kinematic_viscosity
        d_viscosity_d_altitude = atm.partial_kinematic_viscosity_altitude

        omega = inputs["rpm"] * 2.0 * np.pi / 60.0

        partials["reynolds_D", "true_airspeed"] = (
            diameter / viscosity / np.sqrt(true_airspeed**2.0 + (omega * diameter / 2.0) ** 2.0)
        ) * true_airspeed
        partials["reynolds_D", "rpm"] = (
            (diameter / viscosity / np.sqrt(true_airspeed**2.0 + (omega * diameter / 2.0) ** 2.0))
            * (diameter / 2.0) ** 2.0
            * omega
            * 2.0
            * np.pi
            / 60.0
        )
        partials[
            "reynolds_D", "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"
        ] = (
            1.0
            / viscosity
            / np.sqrt((true_airspeed * diameter) ** 2.0 + (omega * diameter**2.0 / 2.0) ** 2.0)
            * (2.0 * true_airspeed**2.0 * diameter + 4.0 * (omega / 2.0) ** 2.0 * diameter**3.0)
            / 2.0
        )
        partials["reynolds_D", "altitude"] = (
            -(
                np.sqrt(true_airspeed**2.0 + (omega * diameter / 2.0) ** 2.0)
                * diameter
                / viscosity**2.0
            )
            * d_viscosity_d_altitude
        )
