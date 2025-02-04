# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import Atmosphere

T0 = Atmosphere(0.0).temperature


class PerformancesMaxPowerITTLimit(om.ExplicitComponent):
    """
    Computation of the design thermodynamic power which would result in the current power
    requirement being limited by the ITT limit.
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
        self.add_input("power_required", units="MW", val=np.nan, shape=number_of_points)
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
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":limit:ITT",
            units="degK",
            val=np.nan,
            desc="Limit ITT of the turboshaft",
        )

        self.add_output(
            "design_power_itt_limit",
            units="MW",
            val=0.5,
            shape=number_of_points,
            desc="Thermodynamic power of the turboshaft at the design point if the ITT was limiting",
        )

        self.declare_partials(
            of="design_power_itt_limit",
            wrt=["mach", "density_ratio", "power_required"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="design_power_itt_limit",
            wrt=[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR",
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t",
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":limit:ITT",
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
        design_t41t = (
            inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t"
            ]
            / T0
        )
        design_opr = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR"
        ]
        limit_itt = (
            inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":limit:ITT"] / T0
        )

        design_power = (
            10 ** (-1.82236)
            * density_ratio
            ** (
                0.63793 * np.log10(design_t41t) ** 2
                + 0.07218 * np.log10(design_opr) * np.log10(limit_itt)
                + 0.26249 * np.log10(density_ratio) * np.log10(power)
                + 0.12410 * np.log10(density_ratio)
                - 1.32058 * np.log10(limit_itt)
            )
            * mach ** (-0.17942 - 0.08544 * np.log10(mach) * np.log10(design_opr))
            * power
            ** (
                -6.21377 * np.log10(design_t41t) ** 2
                + 6.64580 * np.log10(design_opr) * np.log10(limit_itt)
                + 0.93765 * np.log10(power) * np.log10(limit_itt)
                - 0.39916 * np.log10(power) * np.log10(design_opr)
                + 10.42043 * np.log10(design_t41t)
                - 7.55132 * np.log10(limit_itt) ** 2
                - 4.46334 * np.log10(design_opr)
            )
            * design_t41t ** (8.55254 - 4.98088 * np.log10(design_t41t) * np.log10(limit_itt))
            * design_opr
            ** (2.69603 * np.log10(limit_itt) ** 2 - 1.24340 - 0.13934 * np.log10(design_opr) ** 2)
            * limit_itt ** (-3.96490)
        )

        outputs["design_power_itt_limit"] = design_power

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        density_ratio = inputs["density_ratio"]
        mach = inputs["mach"]
        power = inputs["power_required"]
        design_t41t = (
            inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t"
            ]
            / T0
        )
        design_opr = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR"
        ]
        limit_itt = (
            inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":limit:ITT"] / T0
        )

        design_power = (
            10 ** (-1.82236)
            * density_ratio
            ** (
                0.63793 * np.log10(design_t41t) ** 2
                + 0.07218 * np.log10(design_opr) * np.log10(limit_itt)
                + 0.26249 * np.log10(density_ratio) * np.log10(power)
                + 0.12410 * np.log10(density_ratio)
                - 1.32058 * np.log10(limit_itt)
            )
            * mach ** (-0.17942 - 0.08544 * np.log10(mach) * np.log10(design_opr))
            * power
            ** (
                -6.21377 * np.log10(design_t41t) ** 2
                + 6.64580 * np.log10(design_opr) * np.log10(limit_itt)
                + 0.93765 * np.log10(power) * np.log10(limit_itt)
                - 0.39916 * np.log10(power) * np.log10(design_opr)
                + 10.42043 * np.log10(design_t41t)
                - 7.55132 * np.log10(limit_itt) ** 2
                - 4.46334 * np.log10(design_opr)
            )
            * design_t41t ** (8.55254 - 4.98088 * np.log10(design_t41t) * np.log10(limit_itt))
            * design_opr
            ** (2.69603 * np.log10(limit_itt) ** 2 - 1.24340 - 0.13934 * np.log10(design_opr) ** 2)
            * limit_itt ** (-3.96490)
        )

        d_power_d_log_power = 10 ** np.log10(design_power) * np.log(10)

        # Partials derivative for density ratio
        d_log_power_d_log_sigma = (
            0.63793 * np.log10(design_t41t) ** 2
            + 0.07218 * np.log10(design_opr) * np.log10(limit_itt)
            + 2.0 * 0.26249 * np.log10(density_ratio) * np.log10(power)
            + 2.0 * 0.12410 * np.log10(density_ratio)
            - 1.32058 * np.log10(limit_itt)
        )
        d_log_sigma_d_sigma = 1.0 / (np.log(10) * density_ratio)

        partials["design_power_itt_limit", "density_ratio"] = (
            d_power_d_log_power * d_log_power_d_log_sigma * d_log_sigma_d_sigma
        )

        # Partials derivative for mach number
        d_log_power_d_log_mach = -0.17942 - 2.0 * 0.08544 * np.log10(mach) * np.log10(design_opr)
        d_log_mach_d_mach = 1.0 / (np.log(10) * mach)

        partials["design_power_itt_limit", "mach"] = (
            d_power_d_log_power * d_log_power_d_log_mach * d_log_mach_d_mach
        )

        # Partials derivative for shaft power out
        d_log_power_d_log_shaft_power = (
            0.26249 * np.log10(density_ratio) ** 2.0
            - 6.21377 * np.log10(design_t41t) ** 2
            + 6.64580 * np.log10(design_opr) * np.log10(limit_itt)
            + 2.0 * 0.93765 * np.log10(power) * np.log10(limit_itt)
            - 2.0 * 0.39916 * np.log10(power) * np.log10(design_opr)
            + 10.42043 * np.log10(design_t41t)
            - 7.55132 * np.log10(limit_itt) ** 2
            - 4.46334 * np.log10(design_opr)
        )
        d_log_shaft_power_d_shaft_power = 1.0 / (np.log(10) * power)

        partials["design_power_itt_limit", "power_required"] = (
            d_power_d_log_power * d_log_power_d_log_shaft_power * d_log_shaft_power_d_shaft_power
        )

        # Partials derivative for design T41t
        d_log_power_d_log_t41t = (
            2.0 * 0.63793 * np.log10(design_t41t) * np.log10(density_ratio)
            - 2.0 * 6.21377 * np.log10(design_t41t) * np.log10(power)
            + 10.42043 * np.log10(power)
            + 8.55254
            - 2.0 * 4.98088 * np.log10(design_t41t) * np.log10(limit_itt)
        )
        d_log_t41t_d_t41t = 1.0 / (np.log(10) * design_t41t)

        partials[
            "design_power_itt_limit",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t",
        ] = d_power_d_log_power * d_log_power_d_log_t41t * d_log_t41t_d_t41t / T0

        # Partials derivative for design OPR
        d_log_power_d_log_opr_des = (
            0.07218 * np.log10(density_ratio) * np.log10(limit_itt)
            - 0.08544 * np.log10(mach) * np.log10(mach)
            + 6.64580 * np.log10(power) * np.log10(limit_itt)
            - 0.39916 * np.log10(power) ** 2.0
            - 4.46334 * np.log10(power)
            + 2.69603 * np.log10(limit_itt) ** 2
            - 1.24340
            - 3.0 * 0.13934 * np.log10(design_opr) ** 2
        )
        d_log_opr_des_d_opr_des = 1.0 / (np.log(10) * design_opr)

        partials[
            "design_power_itt_limit",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR",
        ] = d_power_d_log_power * d_log_power_d_log_opr_des * d_log_opr_des_d_opr_des

        # Partials derivative for limit ITT
        d_log_power_d_log_itt_limit = (
            0.07218 * np.log10(design_opr) * np.log10(density_ratio)
            - 1.32058 * np.log10(density_ratio)
            + 6.64580 * np.log10(design_opr) * np.log10(power)
            + 0.93765 * np.log10(power) * np.log10(power)
            - 2.0 * 7.55132 * np.log10(limit_itt) * np.log10(power)
            - 4.98088 * np.log10(design_t41t) * np.log10(design_t41t)
            + 2.0 * 2.69603 * np.log10(limit_itt) * np.log10(design_opr)
            - 3.96490
        )
        d_log_itt_limit_d_itt_limit = 1.0 / (np.log(10) * limit_itt)

        partials[
            "design_power_itt_limit",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":limit:ITT",
        ] = d_power_d_log_power * d_log_power_d_log_itt_limit * d_log_itt_limit_d_itt_limit / T0
