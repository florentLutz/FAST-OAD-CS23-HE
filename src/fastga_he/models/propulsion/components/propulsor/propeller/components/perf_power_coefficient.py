# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

# Define here a cutoff thrust coefficient to be used in the case that the thrust coefficient is too
# low. We made the observation that if the thrust coefficient is 0 or very close to zero the power
# coefficient is big enough that it can't be considered nil because of surrogate error.
# This cutoff is an answer to that
CUTOFF_THRUST_COEFFICIENT = 1e-5
KILMER = np.nan


class PerformancesPowerCoefficient(om.ExplicitComponent):
    """
    Computation of the power coefficient of the propeller from the thrust coefficient
    requirement.
    """

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

        self.add_input("thrust_coefficient", val=KILMER, shape=number_of_points)
        self.add_input("tip_mach", val=KILMER, shape=number_of_points)
        self.add_input("reynolds_D", val=KILMER, shape=number_of_points)
        self.add_input("advance_ratio", val=KILMER, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":solidity",
            val=np.nan,
            desc="Solidity of the propeller",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":activity_factor",
            val=np.nan,
            desc="Activity factor of the propeller",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":blade_twist",
            val=np.nan,
            units="rad",
            desc="Twist between the propeller blade root and tip",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":installation_effect",
            val=0.95,
            desc="Increase in the power coefficient due to installation effects on the propeller",
        )

        self.add_output("power_coefficient", shape=number_of_points, val=0.1)

        self.declare_partials(
            of="*",
            wrt=["advance_ratio", "reynolds_D", "tip_mach", "thrust_coefficient"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt=[
                "settings:propulsion:he_power_train:propeller:"
                + propeller_id
                + ":installation_effect",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":blade_twist",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":activity_factor",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":solidity",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        j = inputs["advance_ratio"]
        tip_mach = inputs["tip_mach"]
        re_d = inputs["reynolds_D"]
        solidity = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":solidity"]
        # To avoid warning coming from negative thrust
        ct = np.maximum(
            inputs["thrust_coefficient"],
            np.full_like(inputs["thrust_coefficient"], CUTOFF_THRUST_COEFFICIENT),
        )
        activity_factor = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":activity_factor"
        ]
        twist_blade = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blade_twist"
        ]
        k_installation = inputs[
            "settings:propulsion:he_power_train:propeller:" + propeller_id + ":installation_effect"
        ]

        cp = (
            10**2.43553
            * j
            ** (
                0.61554
                + 0.06980 * np.log10(j) * np.log10(ct) * np.log10(twist_blade)
                - 0.01794 * np.log10(re_d) * np.log10(ct) * np.log10(activity_factor)
                + 0.02595 * np.log10(tip_mach) * np.log10(solidity) * np.log10(ct)
                + 0.00430 * np.log10(j) * np.log10(re_d) ** 2.0
                + 0.09827 * np.log10(ct) * np.log10(twist_blade)
                + 0.03663 * np.log10(solidity) * np.log10(activity_factor)
            )
            * tip_mach ** (-0.00097 * np.log10(re_d) ** 2.0 * np.log10(tip_mach))
            * re_d
            ** (
                -0.35804
                + 0.00018 * np.log10(re_d) ** 3.0
                - 0.01879 * np.log10(ct) ** 3.0
                + 0.00119 * np.log10(re_d) * np.log10(solidity) * np.log10(activity_factor)
                - 0.00886 * np.log10(ct) ** 2.0 * np.log10(twist_blade)
            )
            * solidity
            ** (
                0.08015 * np.log10(ct) ** 2.0 * np.log10(activity_factor)
                + 0.04562 * np.log10(solidity) * np.log10(activity_factor) ** 2.0
                - 0.04121 * np.log10(solidity) * np.log10(ct) * np.log10(twist_blade)
            )
            * ct
            ** (
                1.33164
                + 0.06989 * np.log10(ct) * np.log10(activity_factor)
                + 0.00206 * np.log10(ct) * np.log10(twist_blade) ** 2.0
                + 0.03617 * np.log10(ct) ** 2.0 * np.log10(activity_factor)
            )
        ) / k_installation

        # Let's clip the cp in case the value goes haywire, clipped at zero but if we were to
        # ever look at energy recuperation, it might need to be changed. To compute the upper
        # bound, instead of having a fixed value we will assume a very low efficiency and compute
        # it from there
        lower_efficiency = 0.5
        cp = np.clip(cp, 0.0, j * ct / lower_efficiency)

        # Also as discussed if the ct is low enough (no thrust, the cp will be cutoff)
        cp = np.where(ct > CUTOFF_THRUST_COEFFICIENT, cp, 0.0)

        outputs["power_coefficient"] = cp

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        j = inputs["advance_ratio"]
        tip_mach = inputs["tip_mach"]
        re_d = inputs["reynolds_D"]
        solidity = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":solidity"]
        ct = np.maximum(
            inputs["thrust_coefficient"], np.full_like(inputs["thrust_coefficient"], 1e-4)
        )
        activity_factor = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":activity_factor"
        ]
        twist_blade = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blade_twist"
        ]
        k_installation = inputs[
            "settings:propulsion:he_power_train:propeller:" + propeller_id + ":installation_effect"
        ]

        cp = (
            10**2.43553
            * j
            ** (
                0.61554
                + 0.06980 * np.log10(j) * np.log10(ct) * np.log10(twist_blade)
                - 0.01794 * np.log10(re_d) * np.log10(ct) * np.log10(activity_factor)
                + 0.02595 * np.log10(tip_mach) * np.log10(solidity) * np.log10(ct)
                + 0.00430 * np.log10(j) * np.log10(re_d) ** 2.0
                + 0.09827 * np.log10(ct) * np.log10(twist_blade)
                + 0.03663 * np.log10(solidity) * np.log10(activity_factor)
            )
            * tip_mach ** (-0.00097 * np.log10(re_d) ** 2.0 * np.log10(tip_mach))
            * re_d
            ** (
                -0.35804
                + 0.00018 * np.log10(re_d) ** 3.0
                - 0.01879 * np.log10(ct) ** 3.0
                + 0.00119 * np.log10(re_d) * np.log10(solidity) * np.log10(activity_factor)
                - 0.00886 * np.log10(ct) ** 2.0 * np.log10(twist_blade)
            )
            * solidity
            ** (
                0.08015 * np.log10(ct) ** 2.0 * np.log10(activity_factor)
                + 0.04562 * np.log10(solidity) * np.log10(activity_factor) ** 2.0
                - 0.04121 * np.log10(solidity) * np.log10(ct) * np.log10(twist_blade)
            )
            * ct
            ** (
                1.33164
                + 0.06989 * np.log10(ct) * np.log10(activity_factor)
                + 0.00206 * np.log10(ct) * np.log10(twist_blade) ** 2.0
                + 0.03617 * np.log10(ct) ** 2.0 * np.log10(activity_factor)
            )
        ) / k_installation

        d_pi1_d_log_pi_1 = 10 ** np.log10(cp) * np.log(10)

        d_log_pi1_d_log_pi2 = (
            0.61554
            + 2.0 * 0.06980 * np.log10(j) * np.log10(ct) * np.log10(twist_blade)
            - 0.01794 * np.log10(re_d) * np.log10(ct) * np.log10(activity_factor)
            + 0.02595 * np.log10(tip_mach) * np.log10(solidity) * np.log10(ct)
            + 2.0 * 0.00430 * np.log10(j) * np.log10(re_d) ** 2.0
            + 0.09827 * np.log10(ct) * np.log10(twist_blade)
            + 0.03663 * np.log10(solidity) * np.log10(activity_factor)
        )
        d_log_pi2_d_pi2 = 1.0 / (np.log(10) * j)

        d_log_pi1_d_log_pi3 = +0.02595 * np.log10(j) * np.log10(solidity) * np.log10(
            ct
        ) - 2.0 * 0.00097 * np.log10(re_d) ** 2.0 * np.log10(tip_mach)
        d_log_pi3_d_pi3 = 1.0 / (np.log(10) * tip_mach)

        d_log_pi1_d_log_pi4 = (
            -0.01794 * np.log10(j) * np.log10(ct) * np.log10(activity_factor)
            + 2.0 * 0.00430 * np.log10(j) ** 2.0 * np.log10(re_d)
            - 2.0 * 0.00097 * np.log10(re_d) * np.log10(tip_mach) ** 2.0
            - 0.35804
            + 4.0 * 0.00018 * np.log10(re_d) ** 3.0
            - 0.01879 * np.log10(ct) ** 3.0
            + 2.0 * 0.00119 * np.log10(re_d) * np.log10(solidity) * np.log10(activity_factor)
            - 0.00886 * np.log10(ct) ** 2.0 * np.log10(twist_blade)
        )
        d_log_pi4_d_pi4 = 1.0 / (np.log(10) * re_d)

        d_log_pi1_d_log_pi5 = (
            0.02595 * np.log10(tip_mach) * np.log10(j) * np.log10(ct)
            + 0.03663 * np.log10(j) * np.log10(activity_factor)
            + 0.00119 * np.log10(re_d) * np.log10(re_d) * np.log10(activity_factor)
            + 0.08015 * np.log10(ct) ** 2.0 * np.log10(activity_factor)
            + 2.0 * +0.04562 * np.log10(solidity) * np.log10(activity_factor) ** 2.0
            - 2.0 * 0.04121 * np.log10(solidity) * np.log10(ct) * np.log10(twist_blade)
        )
        d_log_pi5_d_pi5 = 1.0 / (np.log(10) * solidity)

        d_log_pi1_d_log_pi6 = (
            +0.06980 * np.log10(j) ** 2.0 * np.log10(twist_blade)
            - 0.01794 * np.log10(re_d) * np.log10(j) * np.log10(activity_factor)
            + 0.02595 * np.log10(tip_mach) * np.log10(solidity) * np.log10(j)
            + 0.09827 * np.log10(j) * np.log10(twist_blade)
            - 3.0 * 0.01879 * np.log10(ct) ** 2.0 * np.log10(re_d)
            - 2.0 * 0.00886 * np.log10(ct) * np.log10(twist_blade) * np.log10(re_d)
            + 2.0 * 0.08015 * np.log10(ct) * np.log10(activity_factor) * np.log10(solidity)
            - 0.04121 * np.log10(solidity) ** 2.0 * np.log10(twist_blade)
            + 1.33164
            + 2.0 * 0.06989 * np.log10(ct) * np.log10(activity_factor)
            + 2.0 * 0.00206 * np.log10(ct) * np.log10(twist_blade) ** 2.0
            + 3.0 * 0.03617 * np.log10(ct) ** 2.0 * np.log10(activity_factor)
        )
        d_log_pi6_d_pi6 = 1.0 / (np.log(10) * ct)

        d_log_pi1_d_log_pi7 = (
            -0.01794 * np.log10(re_d) * np.log10(ct) * np.log10(j)
            + 0.03663 * np.log10(solidity) * np.log10(j)
            + 0.00119 * np.log10(re_d) * np.log10(solidity) * np.log10(re_d)
            + 0.08015 * np.log10(ct) ** 2.0 * np.log10(solidity)
            + 2.0 * 0.04562 * np.log10(solidity) ** 2.0 * np.log10(activity_factor)
            + 0.06989 * np.log10(ct) ** 2.0
            + 0.03617 * np.log10(ct) ** 3.0
        )
        d_log_pi7_d_pi7 = 1.0 / (np.log(10) * activity_factor)

        d_log_pi1_d_log_pi8 = (
            0.06980 * np.log10(j) * np.log10(ct) * np.log10(j)
            + 0.09827 * np.log10(ct) * np.log10(j)
            - 0.00886 * np.log10(ct) ** 2.0 * np.log10(re_d)
            - 0.04121 * np.log10(solidity) * np.log10(ct) * np.log10(solidity)
            + 2.0 * 0.00206 * np.log10(ct) * np.log10(twist_blade) * np.log10(ct)
        )
        d_log_pi8_d_pi8 = 1.0 / (np.log(10) * twist_blade)

        partials["power_coefficient", "advance_ratio"] = (
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi2 * d_log_pi2_d_pi2
        )
        partials["power_coefficient", "tip_mach"] = (
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi3 * d_log_pi3_d_pi3
        )
        partials["power_coefficient", "reynolds_D"] = (
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi4 * d_log_pi4_d_pi4
        )
        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":solidity",
        ] = d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi5 * d_log_pi5_d_pi5
        partials["power_coefficient", "thrust_coefficient"] = (
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi6 * d_log_pi6_d_pi6
        )
        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":activity_factor",
        ] = d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi7 * d_log_pi7_d_pi7
        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blade_twist",
        ] = d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi8 * d_log_pi8_d_pi8
        partials[
            "power_coefficient",
            "settings:propulsion:he_power_train:propeller:" + propeller_id + ":installation_effect",
        ] = -cp / k_installation
