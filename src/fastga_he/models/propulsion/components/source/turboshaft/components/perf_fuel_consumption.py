# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesTurboshaftFuelConsumption(om.ExplicitComponent):
    """
    Computation of the fuel consumption at each point of the flight.
    """

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
        self.add_input("shaft_power_out", units="kW", val=np.nan, shape=number_of_points)
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
            desc="Ratio of the thermodynamic power divided by the rated power, typical values on the PT6A family is between 1.3 and 2.5",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=np.nan,
            desc="Flat rating of the turboshaft",
        )

        self.add_output("fuel_consumption", units="kg/h", val=120.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        density_ratio = inputs["density_ratio"]
        mach = inputs["mach"]
        power = inputs["shaft_power_out"]
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

        fuel_consumption = (
            10 ** 2.48320
            * density_ratio
            ** (
                2.42792 * np.log10(design_t41t) * np.log10(design_opr)
                - 0.81294 * np.log10(design_power) ** 2
                + 5.25258 * np.log10(design_power)
                - 0.68143 * np.log10(design_t41t) ** 2
                - 2.43656 * np.log10(design_power) * np.log10(design_opr)
                - 4.01243 * np.log10(power)
                + 1.15653 * np.log10(design_power) * np.log10(power)
                + 0.32787 * np.log10(density_ratio)
            )
            * mach ** (-0.10964 - 0.04483 * np.log10(mach) * np.log10(design_opr))
            * design_power
            ** (
                0.02476 * np.log10(power) ** 2
                + 4.82231
                - 0.45400 * np.log10(design_power) * np.log10(design_t41t)
                + 0.71625 * np.log10(design_power)
            )
            * design_t41t ** (0.29604 * np.log10(design_t41t) * np.log10(power) - 2.92384)
            * power ** (-2.76852)
        )

        outputs["fuel_consumption"] = fuel_consumption

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        density_ratio = inputs["density_ratio"]
        mach = inputs["mach"]
        power = inputs["shaft_power_out"]
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

        fuel_consumption = (
            10 ** 2.48320
            * density_ratio
            ** (
                2.42792 * np.log10(design_t41t) * np.log10(design_opr)
                - 0.81294 * np.log10(design_power) ** 2
                + 5.25258 * np.log10(design_power)
                - 0.68143 * np.log10(design_t41t) ** 2
                - 2.43656 * np.log10(design_power) * np.log10(design_opr)
                - 4.01243 * np.log10(power)
                + 1.15653 * np.log10(design_power) * np.log10(power)
                + 0.32787 * np.log10(density_ratio)
            )
            * mach ** (-0.10964 - 0.04483 * np.log10(mach) * np.log10(design_opr))
            * design_power
            ** (
                0.02476 * np.log10(power) ** 2
                + 4.82231
                - 0.45400 * np.log10(design_power) * np.log10(design_t41t)
                + 0.71625 * np.log10(design_power)
            )
            * design_t41t ** (0.29604 * np.log10(design_t41t) * np.log10(power) - 2.92384)
            * power ** (-2.76852)
        )

        d_fc_d_log_fc = 10 ** np.log10(fuel_consumption) * np.log(10)

        # Partials derivative for density ratio
        d_log_fc_d_log_sigma = (
            2.42792 * np.log10(design_t41t) * np.log10(design_opr)
            - 0.81294 * np.log10(design_power) ** 2
            + 5.25258 * np.log10(design_power)
            - 0.68143 * np.log10(design_t41t) ** 2
            - 2.43656 * np.log10(design_power) * np.log10(design_opr)
            - 4.01243 * np.log10(power)
            + 1.15653 * np.log10(design_power) * np.log10(power)
            + 2.0 * 0.32787 * np.log10(density_ratio)
        )
        d_log_sigma_d_sigma = 1.0 / (np.log(10) * density_ratio)

        partials["fuel_consumption", "density_ratio"] = np.diag(
            d_fc_d_log_fc * d_log_fc_d_log_sigma * d_log_sigma_d_sigma
        )

        # Partials derivative for mach number
        d_log_fc_d_log_mach = -0.10964 - 2.0 * 0.04483 * np.log10(mach) * np.log10(design_opr)
        d_log_mach_d_mach = 1.0 / (np.log(10) * mach)

        partials["fuel_consumption", "mach"] = np.diag(
            d_fc_d_log_fc * d_log_fc_d_log_mach * d_log_mach_d_mach
        )

        # Partials derivative for design power related inputs
        d_log_fc_d_log_power_des = (
            -2.0 * 0.81294 * np.log10(design_power) * np.log10(density_ratio)
            + 5.25258 * np.log10(density_ratio)
            - 2.43656 * np.log10(density_ratio) * np.log10(design_opr)
            + 1.15653 * np.log10(density_ratio) * np.log10(power)
            + 0.02476 * np.log10(power) ** 2
            + 4.82231
            - 2.0 * 0.45400 * np.log10(design_power) * np.log10(design_t41t)
            + 2.0 * 0.71625 * np.log10(design_power)
        )
        d_log_power_des_d_power_des = 1.0 / (np.log(10) * design_power)

        partials[
            "fuel_consumption",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = (
            d_fc_d_log_fc * d_log_fc_d_log_power_des * d_log_power_des_d_power_des * power_ratio
        )
        partials[
            "fuel_consumption",
            "data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":design_point:power_ratio",
        ] = (
            d_fc_d_log_fc * d_log_fc_d_log_power_des * d_log_power_des_d_power_des * power_rating
        )

        # Partials derivative for design T41t
        d_log_fc_d_log_t41t = (
            2.42792 * np.log10(density_ratio) * np.log10(design_opr)
            - 2.0 * 0.68143 * np.log10(design_t41t) * np.log10(density_ratio)
            - 0.45400 * np.log10(design_power) ** 2.0
            + 2.0 * 0.29604 * np.log10(design_t41t) * np.log10(power)
            - 2.92384
        )
        d_log_t41t_d_t41t = 1.0 / (np.log(10) * design_t41t)
        partials[
            "fuel_consumption",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t",
        ] = (
            d_fc_d_log_fc * d_log_fc_d_log_t41t * d_log_t41t_d_t41t
        )

        # Partials derivative for design OPR
        d_log_fc_d_log_opr = (
            2.42792 * np.log10(design_t41t) * np.log10(density_ratio)
            - 2.43656 * np.log10(design_power) * np.log10(density_ratio)
            - 0.04483 * np.log10(mach) ** 2.0
        )
        d_log_opr_d_opr = 1.0 / (np.log(10) * design_opr)

        partials[
            "fuel_consumption",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR",
        ] = (
            d_fc_d_log_fc * d_log_fc_d_log_opr * d_log_opr_d_opr
        )

        # Partials derivative for current shaft power
        d_log_fc_d_log_power = (
            -4.01243 * np.log10(density_ratio)
            + 1.15653 * np.log10(design_power) * np.log10(density_ratio)
            + 2.0 * 0.02476 * np.log10(power) * np.log10(design_power)
            + 0.29604 * np.log10(design_t41t) * np.log10(design_t41t)
            - 2.76852
        )
        d_log_power_d_power = 1.0 / (np.log(10) * power)

        partials["fuel_consumption", "shaft_power_out"] = np.diag(
            d_fc_d_log_fc * d_log_fc_d_log_power * d_log_power_d_power
        )
