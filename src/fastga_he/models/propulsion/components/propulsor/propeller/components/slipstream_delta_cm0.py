# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerDeltaCM0(om.ExplicitComponent):
    """
    Compute the increase in profile pitching moment coefficient due to the blowing of the wing,
    taken from :cite:`bouquet:2017`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
            values=["cruise", "takeoff", "landing"],
        )
        self.options.declare(
            "low_speed_aero",
            default=False,
            desc="Boolean to consider low speed aerodynamics",
            types=bool,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        propeller_id = self.options["propeller_id"]
        flaps_position = self.options["flaps_position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_output(
            "delta_Cm0",
            val=-0.01,
            shape=number_of_points,
            desc="Increase in the profile pitching moment coefficient downstream of the propeller",
        )

        self.add_input("data:aerodynamics:wing:" + ls_tag + ":CM0_clean", val=np.nan)
        self.add_input(
            "axial_induction_factor_wing_ac",
            val=np.nan,
            shape=number_of_points,
            desc="Value of the axial induction factor at the wing aerodynamic chord",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":diameter_to_span_ratio",
            val=0.1,
            desc="Diameter of the propeller as a ratio of the wing half span",
        )

        self.declare_partials(
            of="delta_Cm0",
            wrt="axial_induction_factor_wing_ac",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="delta_Cm0",
            wrt=[
                "data:aerodynamics:wing:" + ls_tag + ":CM0_clean",
                "data:propulsion:he_power_train:propeller:"
                + propeller_id
                + ":diameter_to_span_ratio",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

        if flaps_position == "takeoff":
            self.add_input(
                name="data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                val=np.nan,
                desc="Portion of the span, downstream of the propeller, which has flaps",
            )
            self.add_input("data:aerodynamics:flaps:takeoff:CM_2D", val=np.nan)

            self.declare_partials(
                of="delta_Cm0",
                wrt=[
                    "data:aerodynamics:flaps:takeoff:CM_2D",
                    "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                ],
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

        elif flaps_position == "landing":
            self.add_input(
                name="data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                val=np.nan,
                desc="Portion of the span, downstream of the propeller, which has flaps",
            )
            self.add_input("data:aerodynamics:flaps:landing:CM_2D", val=np.nan)

            self.declare_partials(
                of="delta_Cm0",
                wrt=[
                    "data:aerodynamics:flaps:landing:CM_2D",
                    "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                ],
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]
        flaps_position = self.options["flaps_position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        if flaps_position == "takeoff":
            flapped_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
            ]
            delta_cm0_flaps = inputs["data:aerodynamics:flaps:takeoff:CM_2D"] * flapped_ratio

        elif flaps_position == "landing":
            flapped_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
            ]
            delta_cm0_flaps = inputs["data:aerodynamics:flaps:landing:CM_2D"] * flapped_ratio
        else:
            delta_cm0_flaps = 0.0

        cm0_clean = inputs["data:aerodynamics:wing:" + ls_tag + ":CM0_clean"]

        delta_y = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio"
        ]
        a_w = inputs["axial_induction_factor_wing_ac"]

        delta_cm0 = delta_y * a_w**2.0 * (cm0_clean + delta_cm0_flaps)

        outputs["delta_Cm0"] = delta_cm0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]
        flaps_position = self.options["flaps_position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cm0_clean = inputs["data:aerodynamics:wing:" + ls_tag + ":CM0_clean"]

        delta_y = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio"
        ]
        a_w = inputs["axial_induction_factor_wing_ac"]

        if flaps_position == "takeoff":
            flapped_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
            ]
            delta_cm0_2d_flaps = inputs["data:aerodynamics:flaps:takeoff:CM_2D"]
            delta_cm0_flaps = delta_cm0_2d_flaps * flapped_ratio

            partials[
                "delta_Cm0",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
            ] = delta_y * a_w**2.0 * delta_cm0_2d_flaps
            partials["delta_Cm0", "data:aerodynamics:flaps:takeoff:CM_2D"] = (
                delta_y * a_w**2.0 * flapped_ratio
            )

        elif flaps_position == "landing":
            flapped_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
            ]
            delta_cm0_2d_flaps = inputs["data:aerodynamics:flaps:landing:CM_2D"]
            delta_cm0_flaps = delta_cm0_2d_flaps * flapped_ratio

            partials[
                "delta_Cm0",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
            ] = delta_y * a_w**2.0 * delta_cm0_2d_flaps
            partials["delta_Cm0", "data:aerodynamics:flaps:landing:CM_2D"] = (
                delta_y * a_w**2.0 * flapped_ratio
            )

        else:
            delta_cm0_flaps = 0.0

        partials["delta_Cm0", "data:aerodynamics:wing:" + ls_tag + ":CM0_clean"] = (
            delta_y * a_w**2.0
        )
        partials[
            "delta_Cm0",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio",
        ] = a_w**2.0 * (cm0_clean + delta_cm0_flaps)
        partials["delta_Cm0", "axial_induction_factor_wing_ac"] = (
            2.0 * delta_y * a_w * (cm0_clean + delta_cm0_flaps)
        )
