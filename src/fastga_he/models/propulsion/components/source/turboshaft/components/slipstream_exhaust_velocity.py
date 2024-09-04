# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamExhaustVelocity(om.ExplicitComponent):
    """Computation of the velocity of the air at the exhaust of the turboshaft."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]

        self.add_input("mach", val=np.nan, shape=number_of_points)
        self.add_input("density_ratio", val=np.nan, shape=number_of_points)
        self.add_input("power_required", units="kW", val=np.nan, shape=number_of_points)
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR",
            val=np.nan,
            desc="OPR of the turboshaft at the design point",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t",
            units="degK",
            val=np.nan,
            desc="Total temperature at the output of the combustion chamber of the turboshaft "
            "at the design point",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":design_point:power_ratio",
            val=np.nan,
            desc="Ratio of the thermodynamic power divided by the rated power, typical values on "
            "the PT6A family is between 1.3 and 2.5",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=np.nan,
            desc="Flat rating of the turboshaft",
        )

        self.add_output("exhaust_velocity", units="m/s", val=120.0, shape=number_of_points)

        self.declare_partials(
            of="exhaust_velocity",
            wrt=["mach", "density_ratio", "power_required"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="exhaust_velocity",
            wrt=[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR",
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t",
                "data:propulsion:he_power_train:turboshaft:"
                + turboshaft_id
                + ":design_point:power_ratio",
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        density_ratio = inputs["density_ratio"]
        mach = inputs["mach"]
        power = inputs["power_required"]
        # Need to divide by SL temperature for the surrogate
        design_t41t = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t"
        ]
        design_opr = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR"
        ]
        power_rating = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ]
        power_ratio = inputs[
            "data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":design_point:power_ratio"
        ]

        design_power = power_rating * power_ratio

        exhaust_velocity = (
            10**2.45581
            * density_ratio
            ** (
                -0.12369 * np.log10(power) ** 2
                + 0.75124 * np.log10(design_power) * np.log10(power)
                + 1.95855 * np.log10(design_power) * np.log10(design_t41t)
                + 0.08779 * np.log10(density_ratio) * np.log10(design_power)
                + 6.73639 * np.log10(power)
                - 2.85551 * np.log10(design_t41t) * np.log10(power)
                - 4.55507 * np.log10(design_power)
                - 0.48376 * np.log10(design_power) ** 2
            )
            * mach ** (-0.00280 * np.log10(design_power))
            * design_power ** (-0.09904 * np.log10(power))
            * design_t41t
            ** (
                -0.03827 * np.log10(design_t41t) * np.log10(design_opr)
                + 0.03688 * np.log10(design_opr) * np.log10(power)
            )
            * power ** (0.00814 + 0.08323 * np.log10(power))
        )

        outputs["exhaust_velocity"] = exhaust_velocity

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        density_ratio = inputs["density_ratio"]
        mach = inputs["mach"]
        power = inputs["power_required"]
        # Need to divide by SL temperature for the surrogate
        design_t41t = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t"
        ]
        design_opr = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR"
        ]
        power_rating = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ]
        power_ratio = inputs[
            "data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":design_point:power_ratio"
        ]

        design_power = power_rating * power_ratio

        exhaust_velocity = (
            10**2.45581
            * density_ratio
            ** (
                -0.12369 * np.log10(power) ** 2
                + 0.75124 * np.log10(design_power) * np.log10(power)
                + 1.95855 * np.log10(design_power) * np.log10(design_t41t)
                + 0.08779 * np.log10(density_ratio) * np.log10(design_power)
                + 6.73639 * np.log10(power)
                - 2.85551 * np.log10(design_t41t) * np.log10(power)
                - 4.55507 * np.log10(design_power)
                - 0.48376 * np.log10(design_power) ** 2
            )
            * mach ** (-0.00280 * np.log10(design_power))
            * design_power ** (-0.09904 * np.log10(power))
            * design_t41t
            ** (
                -0.03827 * np.log10(design_t41t) * np.log10(design_opr)
                + 0.03688 * np.log10(design_opr) * np.log10(power)
            )
            * power ** (0.00814 + 0.08323 * np.log10(power))
        )

        d_v8_d_log_v8 = 10 ** np.log10(exhaust_velocity) * np.log(10)

        # Partials derivative for density ratio
        d_log_v8_d_log_sigma = (
            -0.12369 * np.log10(power) ** 2
            + 0.75124 * np.log10(design_power) * np.log10(power)
            + 1.95855 * np.log10(design_power) * np.log10(design_t41t)
            + 2.0 * 0.08779 * np.log10(density_ratio) * np.log10(design_power)
            + 6.73639 * np.log10(power)
            - 2.85551 * np.log10(design_t41t) * np.log10(power)
            - 4.55507 * np.log10(design_power)
            - 0.48376 * np.log10(design_power) ** 2
        )
        d_log_sigma_d_sigma = 1.0 / (np.log(10) * density_ratio)

        partials["exhaust_velocity", "density_ratio"] = (
            d_v8_d_log_v8 * d_log_v8_d_log_sigma * d_log_sigma_d_sigma
        )

        # Partials derivative for mach number
        d_log_v8_d_log_mach = -0.00280 * np.log10(design_power)
        d_log_mach_d_mach = 1.0 / (np.log(10) * mach)

        partials["exhaust_velocity", "mach"] = (
            d_v8_d_log_v8 * d_log_v8_d_log_mach * d_log_mach_d_mach
        )

        # Partials derivative for design power related inputs
        d_log_v8_d_log_power_des = (
            0.75124 * np.log10(density_ratio) * np.log10(power)
            + 1.95855 * np.log10(density_ratio) * np.log10(design_t41t)
            + 0.08779 * np.log10(density_ratio) ** 2.0
            - 4.55507 * np.log10(density_ratio)
            - 2.0 * 0.48376 * np.log10(design_power) * np.log10(density_ratio)
            - 0.00280 * np.log10(mach)
            - 0.09904 * np.log10(power)
        )
        d_log_power_des_d_power_des = 1.0 / (np.log(10) * design_power)

        partials[
            "exhaust_velocity",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = d_v8_d_log_v8 * d_log_v8_d_log_power_des * d_log_power_des_d_power_des * power_ratio
        partials[
            "exhaust_velocity",
            "data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":design_point:power_ratio",
        ] = d_v8_d_log_v8 * d_log_v8_d_log_power_des * d_log_power_des_d_power_des * power_rating

        # Partials derivative for design T41t
        d_log_v8_d_log_t41t = (
            1.95855 * np.log10(design_power) * np.log10(density_ratio)
            - 2.85551 * np.log10(density_ratio) * np.log10(power)
            - 2.0 * 0.03827 * np.log10(design_t41t) * np.log10(design_opr)
            + 0.03688 * np.log10(design_opr) * np.log10(power)
        )
        d_log_t41t_d_t41t = 1.0 / (np.log(10) * design_t41t)
        partials[
            "exhaust_velocity",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t",
        ] = d_v8_d_log_v8 * d_log_v8_d_log_t41t * d_log_t41t_d_t41t

        # Partials derivative for design OPR
        d_log_v8_d_log_opr = -0.03827 * np.log10(design_t41t) * np.log10(
            design_t41t
        ) + 0.03688 * np.log10(design_t41t) * np.log10(power)
        d_log_opr_d_opr = 1.0 / (np.log(10) * design_opr)

        partials[
            "exhaust_velocity",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR",
        ] = d_v8_d_log_v8 * d_log_v8_d_log_opr * d_log_opr_d_opr

        # Partials derivative for current shaft power
        d_log_v8_d_log_power = (
            -2.0 * 0.12369 * np.log10(power) * np.log10(density_ratio)
            + 0.75124 * np.log10(design_power) * np.log10(density_ratio)
            + 6.73639 * np.log10(density_ratio)
            - 2.85551 * np.log10(design_t41t) * np.log10(density_ratio)
            - 0.09904 * np.log10(design_power)
            + 0.03688 * np.log10(design_opr) * np.log10(design_t41t)
            + 0.00814
            + 2.0 * 0.08323 * np.log10(power)
        )
        d_log_power_d_power = 1.0 / (np.log(10) * power)

        partials["exhaust_velocity", "power_required"] = (
            d_v8_d_log_v8 * d_log_v8_d_log_power * d_log_power_d_power
        )
