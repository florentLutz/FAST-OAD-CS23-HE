# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerDeltaCD0(om.ExplicitComponent):
    """
    Compute the increase in profile drag coefficient due to the blowing of the wing, taken from
    :cite:`biber:2011` and adapted in :cite:`de:2016`. Will be the only contribution due to
    propeller slipstream effects because the lift induced drag term is computed on its own.
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

    def setup(self):
        number_of_points = self.options["number_of_points"]
        propeller_id = self.options["propeller_id"]
        flaps_position = self.options["flaps_position"]

        self.add_output(
            "delta_Cd",
            val=0.01,
            shape=number_of_points,
            desc="Increase in the profile drag coefficient downstream of the propeller",
        )

        self.add_input("data:geometry:wing:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:wing:cruise:CD0", val=np.nan)
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
            of="delta_Cd",
            wrt="axial_induction_factor_wing_ac",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="delta_Cd",
            wrt=[
                "data:geometry:wing:wet_area",
                "data:geometry:wing:area",
                "data:aerodynamics:wing:cruise:CD0",
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
            self.add_input("data:aerodynamics:flaps:takeoff:CD_2D", val=np.nan)
            self.declare_partials(
                of="delta_Cd",
                wrt=[
                    "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                    "data:aerodynamics:flaps:takeoff:CD_2D",
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
            self.add_input("data:aerodynamics:flaps:landing:CD_2D", val=np.nan)
            self.declare_partials(
                of="delta_Cd",
                wrt=[
                    "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                    "data:aerodynamics:flaps:landing:CD_2D",
                ],
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]
        flaps_position = self.options["flaps_position"]

        if flaps_position == "takeoff":
            flapped_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
            ]
            delta_cd0_flaps = inputs["data:aerodynamics:flaps:takeoff:CD_2D"] * flapped_ratio

        elif flaps_position == "landing":
            flapped_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
            ]
            delta_cd0_flaps = inputs["data:aerodynamics:flaps:landing:CD_2D"] * flapped_ratio
        else:
            delta_cd0_flaps = 0.0

        wing_wet_area = inputs["data:geometry:wing:wet_area"]
        wing_dry_area = inputs["data:geometry:wing:area"]  # LOL
        cd0 = inputs["data:aerodynamics:wing:cruise:CD0"]
        cf = cd0 * wing_dry_area / wing_wet_area

        delta_y = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio"
        ]
        a_w = inputs["axial_induction_factor_wing_ac"]

        delta_cd0 = delta_y * a_w**2.0 * (cf + delta_cd0_flaps)

        outputs["delta_Cd"] = delta_cd0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]
        flaps_position = self.options["flaps_position"]

        wing_wet_area = inputs["data:geometry:wing:wet_area"]
        wing_dry_area = inputs["data:geometry:wing:area"]  # LOL
        cd0 = inputs["data:aerodynamics:wing:cruise:CD0"]

        delta_y = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio"
        ]
        a_w = inputs["axial_induction_factor_wing_ac"]

        if flaps_position == "takeoff":
            flapped_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
            ]
            delta_cd0_2d_flaps = inputs["data:aerodynamics:flaps:takeoff:CD_2D"]
            delta_cd0_flaps = delta_cd0_2d_flaps * flapped_ratio

            partials[
                "delta_Cd",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
            ] = delta_y * a_w**2.0 * delta_cd0_2d_flaps
            partials["delta_Cd", "data:aerodynamics:flaps:takeoff:CD_2D"] = (
                delta_y * a_w**2.0 * flapped_ratio
            )

        elif flaps_position == "landing":
            flapped_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
            ]
            delta_cd0_2d_flaps = inputs["data:aerodynamics:flaps:landing:CD_2D"]
            delta_cd0_flaps = delta_cd0_2d_flaps * flapped_ratio

            partials[
                "delta_Cd",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
            ] = delta_y * a_w**2.0 * delta_cd0_2d_flaps
            partials["delta_Cd", "data:aerodynamics:flaps:landing:CD_2D"] = (
                delta_y * a_w**2.0 * flapped_ratio
            )

        else:
            delta_cd0_flaps = 0.0

        partials["delta_Cd", "data:geometry:wing:wet_area"] = (
            -delta_y * a_w**2.0 * cd0 * wing_dry_area / wing_wet_area**2.0
        )
        partials["delta_Cd", "data:geometry:wing:area"] = delta_y * a_w**2.0 * cd0 / wing_wet_area
        partials["delta_Cd", "data:aerodynamics:wing:cruise:CD0"] = (
            delta_y * a_w**2.0 * wing_dry_area / wing_wet_area
        )
        partials[
            "delta_Cd",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio",
        ] = a_w**2.0 * (cd0 * wing_dry_area / wing_wet_area + delta_cd0_flaps)
        partials["delta_Cd", "axial_induction_factor_wing_ac"] = (
            2.0 * delta_y * a_w * (cd0 * wing_dry_area / wing_wet_area + delta_cd0_flaps)
        )
