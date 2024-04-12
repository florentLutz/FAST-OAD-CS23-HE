# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import Atmosphere

T0 = Atmosphere(0.0).temperature


class PerformancesMaxPowerOPRLimit(om.ExplicitComponent):
    """
    Computation of the design thermodynamic power which would result in the current power
    requirement being limited by the OPR limit.
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
        self.add_input("shaft_power_out", units="MW", val=np.nan, shape=number_of_points)
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
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":limit:OPR",
            val=np.nan,
            desc="Limit OPR of the turboshaft",
        )

        self.add_output(
            "design_power_opr_limit",
            units="MW",
            val=1500.0,
            shape=number_of_points,
            desc="Thermodynamic power of the turboshaft at the design point if the OPR was limiting",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        density_ratio = inputs["density_ratio"]
        mach = inputs["mach"]
        power = inputs["shaft_power_out"]
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
        limit_opr = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":limit:OPR"
        ]

        design_power = (
            10 ** 1.09582
            * density_ratio
            ** (3.80751 * np.log10(design_t41t) ** 2 - 4.53509 * np.log10(design_t41t))
            * mach
            ** (
                -1.94092
                + 0.90357 * np.log10(mach) * np.log10(design_t41t)
                - 1.84547 * np.log10(mach)
                - 0.43545 * np.log10(mach) ** 2
                + 1.26038 * np.log10(design_t41t)
                - 0.64014 * np.log10(design_opr)
                + 0.78747 * np.log10(design_t41t) * np.log10(limit_opr)
            )
            * power ** (3.08834 * np.log10(design_t41t) - 2.37741 * np.log10(design_t41t) ** 2)
            * design_t41t
            ** (
                -11.83051
                + 21.00196 * np.log10(design_t41t) ** 2
                + 13.67732 * np.log10(limit_opr) ** 2
                - 29.45429 * np.log10(design_t41t) * np.log10(design_opr)
                + 31.43437 * np.log10(design_opr)
                - 16.09090 * np.log10(design_t41t) * np.log10(limit_opr)
            )
            * design_opr ** (-6.22319)
            * limit_opr ** (-9.27983 * np.log10(limit_opr) + 5.82076)
        )

        outputs["design_power_opr_limit"] = design_power

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        density_ratio = inputs["density_ratio"]
        mach = inputs["mach"]
        power = inputs["shaft_power_out"]
        design_t41t = (
            inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t"
            ]
            / T0
        )
        design_opr = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR"
        ]
        limit_opr = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":limit:OPR"
        ]

        design_power = (
            10 ** 1.09582
            * density_ratio
            ** (3.80751 * np.log10(design_t41t) ** 2 - 4.53509 * np.log10(design_t41t))
            * mach
            ** (
                -1.94092
                + 0.90357 * np.log10(mach) * np.log10(design_t41t)
                - 1.84547 * np.log10(mach)
                - 0.43545 * np.log10(mach) ** 2
                + 1.26038 * np.log10(design_t41t)
                - 0.64014 * np.log10(design_opr)
                + 0.78747 * np.log10(design_t41t) * np.log10(limit_opr)
            )
            * power ** (3.08834 * np.log10(design_t41t) - 2.37741 * np.log10(design_t41t) ** 2)
            * design_t41t
            ** (
                -11.83051
                + 21.00196 * np.log10(design_t41t) ** 2
                + 13.67732 * np.log10(limit_opr) ** 2
                - 29.45429 * np.log10(design_t41t) * np.log10(design_opr)
                + 31.43437 * np.log10(design_opr)
                - 16.09090 * np.log10(design_t41t) * np.log10(limit_opr)
            )
            * design_opr ** (-6.22319)
            * limit_opr ** (-9.27983 * np.log10(limit_opr) + 5.82076)
        )

        d_power_d_log_power = 10 ** np.log10(design_power) * np.log(10)

        # Partials derivative for density ratio
        d_log_power_d_log_sigma = 3.80751 * np.log10(design_t41t) ** 2 - 4.53509 * np.log10(
            design_t41t
        )
        d_log_sigma_d_sigma = 1.0 / (np.log(10) * density_ratio)

        partials["design_power_opr_limit", "density_ratio"] = np.diag(
            d_power_d_log_power * d_log_power_d_log_sigma * d_log_sigma_d_sigma
        )

        # Partials derivative for mach number
        d_log_power_d_log_mach = (
            -1.94092
            + 2.0 * 0.90357 * np.log10(mach) * np.log10(design_t41t)
            - 2.0 * 1.84547 * np.log10(mach)
            - 3.0 * 0.43545 * np.log10(mach) ** 2
            + 1.26038 * np.log10(design_t41t)
            - 0.64014 * np.log10(design_opr)
            + 0.78747 * np.log10(design_t41t) * np.log10(limit_opr)
        )
        d_log_mach_d_mach = 1.0 / (np.log(10) * mach)

        partials["design_power_opr_limit", "mach"] = np.diag(
            d_power_d_log_power * d_log_power_d_log_mach * d_log_mach_d_mach
        )

        # Partials derivative for shaft power out
        d_log_power_d_log_shaft_power = (
            3.08834 * np.log10(design_t41t) - 2.37741 * np.log10(design_t41t) ** 2
        )
        d_log_shaft_power_d_shaft_power = 1.0 / (np.log(10) * power)

        partials["design_power_opr_limit", "shaft_power_out"] = np.diag(
            d_power_d_log_power * d_log_power_d_log_shaft_power * d_log_shaft_power_d_shaft_power
        )

        # Partials derivative for design T41t
        d_log_power_d_log_t41t = (
            2.0 * 3.80751 * np.log10(design_t41t) * np.log10(density_ratio)
            - 4.53509 * np.log10(density_ratio)
            + 0.90357 * np.log10(mach) ** 2.0
            + 1.26038 * np.log10(mach)
            + 0.78747 * np.log10(mach) * np.log10(limit_opr)
            + 3.08834 * np.log10(power)
            - 2.0 * 2.37741 * np.log10(design_t41t) * np.log10(power)
            - 11.83051
            + 3.0 * 21.00196 * np.log10(design_t41t) ** 2
            + 13.67732 * np.log10(limit_opr) ** 2
            - 2.0 * 29.45429 * np.log10(design_t41t) * np.log10(design_opr)
            + 31.43437 * np.log10(design_opr)
            - 2.0 * 16.09090 * np.log10(design_t41t) * np.log10(limit_opr)
        )
        d_log_t41t_d_t41t = 1.0 / (np.log(10) * design_t41t)

        partials[
            "design_power_opr_limit",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t",
        ] = (
            d_power_d_log_power * d_log_power_d_log_t41t * d_log_t41t_d_t41t / T0
        )

        # Partials derivative for design OPR
        d_log_power_d_log_opr_des = (
            -0.64014 * np.log10(mach)
            - 29.45429 * np.log10(design_t41t) ** 2.0
            + 31.43437 * np.log10(design_t41t)
            - 6.22319
        )
        d_log_opr_des_d_opr_des = 1.0 / (np.log(10) * design_opr)

        partials[
            "design_power_opr_limit",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR",
        ] = (
            d_power_d_log_power * d_log_power_d_log_opr_des * d_log_opr_des_d_opr_des
        )

        # Partials derivative for limit OPR
        d_log_power_d_log_opr_limit = (
            +0.78747 * np.log10(design_t41t) * np.log10(mach)
            + 2.0 * 13.67732 * np.log10(limit_opr) * np.log10(design_t41t)
            - 16.09090 * np.log10(design_t41t) * np.log10(design_t41t)
            + 5.82076
            - 2.0 * 9.27983 * np.log10(limit_opr)
        )
        d_log_opr_limit_d_opr_limit = 1.0 / (np.log(10) * limit_opr)

        partials[
            "design_power_opr_limit",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":limit:OPR",
        ] = (
            d_power_d_log_power * d_log_power_d_log_opr_limit * d_log_opr_limit_d_opr_limit
        )
