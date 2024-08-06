# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesLosses(om.ExplicitComponent):
    """
    Computation of the generator losses from torque and rpm. Same model as in the AFPMSM.

    Main losses considered in this model are :
    - Joules losses (alpha * T^2).
    - Hysteresis losses (beta * omega).
    - Eddy current losses (gamma * omega^2).
    """

    def initialize(self):
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        generator_id = self.options["generator_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("torque_in", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_input(
            name="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":loss_coefficient:alpha",
            val=np.nan,
            units="W/N**2/m**2",
        )
        self.add_input(
            name="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":loss_coefficient:beta",
            val=np.nan,
            units="W*s/rad",
        )
        self.add_input(
            name="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":loss_coefficient:gamma",
            val=np.nan,
            units="W*s**2/rad**2",
        )

        self.add_output("power_losses", units="W", val=0.0, shape=number_of_points)

        self.declare_partials(
            of="power_losses",
            wrt=["torque_in", "rpm"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="power_losses",
            wrt=[
                "data:propulsion:he_power_train:generator:"
                + generator_id
                + ":loss_coefficient:gamma",
                "data:propulsion:he_power_train:generator:"
                + generator_id
                + ":loss_coefficient:beta",
                "data:propulsion:he_power_train:generator:"
                + generator_id
                + ":loss_coefficient:alpha",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]

        torque = inputs["torque_in"]
        rpm = inputs["rpm"]

        omega = rpm * 2.0 * np.pi / 60.0

        alpha = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:alpha"
        ]
        beta = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:beta"
        ]
        gamma = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:gamma"
        ]

        power_losses = alpha * torque**2.0 + beta * omega + gamma * omega**2.0

        outputs["power_losses"] = power_losses

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        generator_id = self.options["generator_id"]

        torque = inputs["torque_in"]
        rpm = inputs["rpm"]

        omega = rpm * 2.0 * np.pi / 60.0

        alpha = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:alpha"
        ]
        beta = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:beta"
        ]
        gamma = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:gamma"
        ]

        partials["power_losses", "torque_in"] = 2.0 * alpha * torque
        partials["power_losses", "rpm"] = beta + 2.0 * gamma * omega * 2.0 * np.pi / 60.0
        partials[
            "power_losses",
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:alpha",
        ] = torque**2.0
        partials[
            "power_losses",
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:beta",
        ] = omega
        partials[
            "power_losses",
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:gamma",
        ] = omega**2.0
