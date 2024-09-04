# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamExhaustMassFlow(om.ExplicitComponent):
    """Computation of the air mass flow at the exhaust of the turboshaft."""

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
            desc="Ratio of the thermodynamic power divided by the rated power, typical values on the PT6A family is between 1.3 and 2.5",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=np.nan,
            desc="Flat rating of the turboshaft",
        )

        self.add_output("exhaust_mass_flow", units="kg/s", val=10.0, shape=number_of_points)

        self.declare_partials(
            of="exhaust_mass_flow",
            wrt=["density_ratio", "power_required"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="exhaust_mass_flow",
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

        exhaust_mass_flow = (
            10**2.26440
            * density_ratio
            ** (-1.11652 * np.log10(design_t41t) - 0.00080 * np.log10(design_power) + 4.25022)
            * design_power
            ** (5.93864 - 0.92386 * np.log10(design_power) + 0.70938 * np.log10(design_opr))
            * design_t41t ** (-3.63334 - 0.80058 * np.log10(design_opr))
            * design_opr ** (0.21593 * np.log10(power))
        )

        outputs["exhaust_mass_flow"] = exhaust_mass_flow

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        density_ratio = inputs["density_ratio"]
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

        exhaust_mass_flow = (
            10**2.26440
            * density_ratio
            ** (-1.11652 * np.log10(design_t41t) - 0.00080 * np.log10(design_power) + 4.25022)
            * design_power
            ** (5.93864 - 0.92386 * np.log10(design_power) + 0.70938 * np.log10(design_opr))
            * design_t41t ** (-3.63334 - 0.80058 * np.log10(design_opr))
            * design_opr ** (0.21593 * np.log10(power))
        )

        d_m_dot_8_d_log_m_dot_8 = 10 ** np.log10(exhaust_mass_flow) * np.log(10)

        # Partials derivative for density ratio
        d_log_m_dot_8_d_log_sigma = (
            -1.11652 * np.log10(design_t41t) - 0.00080 * np.log10(design_power) + 4.25022
        )
        d_log_sigma_d_sigma = 1.0 / (np.log(10) * density_ratio)

        partials["exhaust_mass_flow", "density_ratio"] = (
            d_m_dot_8_d_log_m_dot_8 * d_log_m_dot_8_d_log_sigma * d_log_sigma_d_sigma
        )

        # Partials derivative for design power related inputs
        d_log_m_dot_8_d_log_power_des = (
            -0.00080 * np.log10(density_ratio)
            + 5.93864
            - 2.0 * 0.92386 * np.log10(design_power)
            + 0.70938 * np.log10(design_opr)
        )
        d_log_power_des_d_power_des = 1.0 / (np.log(10) * design_power)

        partials[
            "exhaust_mass_flow",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = (
            d_m_dot_8_d_log_m_dot_8
            * d_log_m_dot_8_d_log_power_des
            * d_log_power_des_d_power_des
            * power_ratio
        )
        partials[
            "exhaust_mass_flow",
            "data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":design_point:power_ratio",
        ] = (
            d_m_dot_8_d_log_m_dot_8
            * d_log_m_dot_8_d_log_power_des
            * d_log_power_des_d_power_des
            * power_rating
        )

        # Partials derivative for design T41t
        d_log_m_dot_8_d_log_t41t = (
            -1.11652 * np.log10(density_ratio) - 3.63334 - 0.80058 * np.log10(design_opr)
        )
        d_log_t41t_d_t41t = 1.0 / (np.log(10) * design_t41t)
        partials[
            "exhaust_mass_flow",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:T41t",
        ] = d_m_dot_8_d_log_m_dot_8 * d_log_m_dot_8_d_log_t41t * d_log_t41t_d_t41t

        # Partials derivative for design OPR
        d_log_m_dot_8_d_log_opr = (
            0.70938 * np.log10(design_power)
            - 0.80058 * np.log10(design_t41t)
            + 0.21593 * np.log10(power)
        )
        d_log_opr_d_opr = 1.0 / (np.log(10) * design_opr)

        partials[
            "exhaust_mass_flow",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":design_point:OPR",
        ] = d_m_dot_8_d_log_m_dot_8 * d_log_m_dot_8_d_log_opr * d_log_opr_d_opr

        # Partials derivative for current shaft power
        d_log_m_dot_8_d_log_power = 0.21593 * np.log10(design_opr)
        d_log_power_d_power = 1.0 / (np.log(10) * power)

        partials["exhaust_mass_flow", "power_required"] = (
            d_m_dot_8_d_log_m_dot_8 * d_log_m_dot_8_d_log_power * d_log_power_d_power
        )
