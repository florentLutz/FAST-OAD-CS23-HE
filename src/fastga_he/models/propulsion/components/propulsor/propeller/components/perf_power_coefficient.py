# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

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

        self.add_output("power_coefficient", shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        j = inputs["advance_ratio"]
        tip_mach = inputs["tip_mach"]
        re_d = inputs["reynolds_D"]
        solidity = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":solidity"]
        # To avoid warning coming from negative thrust
        ct = np.maximum(
            inputs["thrust_coefficient"], np.full_like(inputs["thrust_coefficient"], 1e-4)
        )
        activity_factor = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":activity_factor"
        ]
        twist_blade = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blade_twist"
        ]

        cp = (
            10 ** 2.31538
            * j
            ** (
                +0.05414 * np.log10(re_d)
                + 0.06795 * np.log10(j) * np.log10(activity_factor) ** 2
                - 0.47030 * np.log10(ct)
                - 0.05973 * np.log10(j) ** 2
                + 0.17931 * np.log10(ct) * np.log10(activity_factor) * np.log10(twist_blade)
                - 0.19079 * np.log10(twist_blade) ** 2
                - 0.07782 * np.log10(j) * np.log10(activity_factor) * np.log10(twist_blade)
                - 0.07683 * np.log10(ct) ** 2 * np.log10(activity_factor)
            )
            * tip_mach ** (+0.00011 * np.log10(re_d) ** 3)
            * re_d
            ** (
                +0.02100 * np.log10(solidity) * np.log10(ct) ** 2
                - 0.32454
                - 0.00078 * np.log10(ct) ** 2 * np.log10(twist_blade)
                + 0.00371 * np.log10(re_d) * np.log10(ct) ** 2
                + 0.00015 * np.log10(re_d) ** 3
            )
            * solidity ** (-0.02698 * np.log10(solidity) ** 2 * np.log10(activity_factor))
            * ct
            ** (
                +1.48237
                + 0.01548 * np.log10(ct) ** 2 * np.log10(activity_factor)
                + 0.10762 * np.log10(ct)
                - 0.05141 * np.log10(ct) ** 2
            )
            * activity_factor ** (-0.00292 * np.log10(activity_factor) ** 2 * np.log10(twist_blade))
        )

        outputs["power_coefficient"] = cp

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        j = inputs["advance_ratio"]
        tip_mach = inputs["tip_mach"]
        re_d = inputs["reynolds_D"]
        solidity = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":solidity"]
        ct = inputs["thrust_coefficient"]
        activity_factor = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":activity_factor"
        ]
        twist_blade = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blade_twist"
        ]

        cp = (
            10 ** 2.31538
            * j
            ** (
                +0.05414 * np.log10(re_d)
                + 0.06795 * np.log10(j) * np.log10(activity_factor) ** 2
                - 0.47030 * np.log10(ct)
                - 0.05973 * np.log10(j) ** 2
                + 0.17931 * np.log10(ct) * np.log10(activity_factor) * np.log10(twist_blade)
                - 0.19079 * np.log10(twist_blade) ** 2
                - 0.07782 * np.log10(j) * np.log10(activity_factor) * np.log10(twist_blade)
                - 0.07683 * np.log10(ct) ** 2 * np.log10(activity_factor)
            )
            * tip_mach ** (+0.00011 * np.log10(re_d) ** 3)
            * re_d
            ** (
                +0.02100 * np.log10(solidity) * np.log10(ct) ** 2
                - 0.32454
                - 0.00078 * np.log10(ct) ** 2 * np.log10(twist_blade)
                + 0.00371 * np.log10(re_d) * np.log10(ct) ** 2
                + 0.00015 * np.log10(re_d) ** 3
            )
            * solidity ** (-0.02698 * np.log10(solidity) ** 2 * np.log10(activity_factor))
            * ct
            ** (
                +1.48237
                + 0.01548 * np.log10(ct) ** 2 * np.log10(activity_factor)
                + 0.10762 * np.log10(ct)
                - 0.05141 * np.log10(ct) ** 2
            )
            * activity_factor ** (-0.00292 * np.log10(activity_factor) ** 2 * np.log10(twist_blade))
        )

        d_pi1_d_log_pi_1 = 10 ** np.log10(cp) * np.log(10)

        d_log_pi1_d_log_pi2 = (
            0.05414 * np.log10(re_d)
            + 2.0 * 0.06795 * np.log10(j) * np.log10(activity_factor) ** 2
            - 0.47030 * np.log10(ct)
            - 3.0 * 0.05973 * np.log10(j) ** 2
            + 0.17931 * np.log10(ct) * np.log10(activity_factor) * np.log10(twist_blade)
            - 0.19079 * np.log10(twist_blade) ** 2
            - 2.0 * 0.07782 * np.log10(j) * np.log10(activity_factor) * np.log10(twist_blade)
            - 0.07683 * np.log10(ct) ** 2 * np.log10(activity_factor)
        )
        d_log_pi2_d_pi2 = 1.0 / (np.log(10) * j)

        d_log_pi1_d_log_pi3 = +0.00011 * np.log10(re_d) ** 3
        d_log_pi3_d_pi3 = 1.0 / (np.log(10) * tip_mach)

        d_log_pi1_d_log_pi4 = (
            0.00033 * np.log10(tip_mach) * np.log10(re_d) ** 2.0
            + 0.00742 * np.log10(ct) ** 2.0 * np.log10(re_d)
            + 0.021 * np.log10(solidity) * np.log10(ct) ** 2.0
            + 0.05414 * np.log10(j)
            - 0.00078 * np.log10(ct) ** 2.0 * np.log10(twist_blade)
            + 0.0006 * np.log10(re_d) ** 3.0
            - 0.32454
        )
        d_log_pi4_d_pi4 = 1.0 / (np.log(10) * re_d)

        d_log_pi1_d_log_pi5 = 0.02100 * np.log10(re_d) * np.log10(
            ct
        ) ** 2 - 3.0 * 0.02698 * np.log10(solidity) ** 2.0 * np.log10(activity_factor)
        d_log_pi5_d_pi5 = 1.0 / (np.log(10) * solidity)

        d_log_pi1_d_log_pi6 = (
            -0.47030 * np.log10(j)
            + 0.17931 * np.log10(j) * np.log10(activity_factor) * np.log10(twist_blade)
            - 2.0 * 0.07683 * np.log10(ct) * np.log10(j) * np.log10(activity_factor)
            + 2.0 * 0.02100 * np.log10(solidity) * np.log10(ct) * np.log10(re_d)
            - 2.0 * 0.00078 * np.log10(twist_blade) * np.log10(ct) * np.log10(re_d)
            + 2.0 * 0.00371 * np.log10(ct) * np.log10(re_d) ** 2.0
            + 1.48237
            + 3.0 * 0.01548 * np.log10(ct) ** 2 * np.log10(activity_factor)
            + 2.0 * 0.10762 * np.log10(ct)
            - 3.0 * 0.05141 * np.log10(ct) ** 2
        )
        d_log_pi6_d_pi6 = 1.0 / (np.log(10) * ct)

        d_log_pi1_d_log_pi7 = (
            2.0 * 0.06795 * np.log10(j) ** 2.0 * np.log10(activity_factor)
            + 0.17931 * np.log10(ct) * np.log10(j) * np.log10(twist_blade)
            - 0.07782 * np.log10(j) ** 2.0 * np.log10(twist_blade)
            - 0.07683 * np.log10(ct) ** 2 * np.log10(j)
            - 0.02698 * np.log10(solidity) ** 3
            + 0.01548 * np.log10(ct) ** 3
            - 3.0 * 0.00292 * np.log10(activity_factor) ** 2 * np.log10(twist_blade)
        )
        d_log_pi7_d_pi7 = 1.0 / (np.log(10) * activity_factor)

        d_log_pi1_d_log_pi8 = (
            +0.17931 * np.log10(ct) * np.log10(activity_factor) * np.log10(j)
            - 2.0 * 0.19079 * np.log10(twist_blade) * np.log10(j)
            - 0.07782 * np.log10(j) * np.log10(activity_factor) * np.log10(j)
            - 0.00078 * np.log10(ct) ** 2 * np.log10(re_d)
            - 0.00292 * np.log10(activity_factor) ** 3
        )
        d_log_pi8_d_pi8 = 1.0 / (np.log(10) * twist_blade)

        partials["power_coefficient", "advance_ratio"] = np.diag(
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi2 * d_log_pi2_d_pi2
        )
        partials["power_coefficient", "tip_mach"] = np.diag(
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi3 * d_log_pi3_d_pi3
        )
        partials["power_coefficient", "reynolds_D"] = np.diag(
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi4 * d_log_pi4_d_pi4
        )
        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":solidity",
        ] = (
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi5 * d_log_pi5_d_pi5
        )
        partials[
            "power_coefficient",
            "thrust_coefficient",
        ] = np.diag(d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi6 * d_log_pi6_d_pi6)
        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":activity_factor",
        ] = (
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi7 * d_log_pi7_d_pi7
        )
        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blade_twist",
        ] = (
            d_pi1_d_log_pi_1 * d_log_pi1_d_log_pi8 * d_log_pi8_d_pi8
        )
