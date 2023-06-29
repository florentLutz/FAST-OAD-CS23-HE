# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerDeltaCMAlpha(om.ExplicitComponent):
    """
    Compute the increase in pitching moment about the quarter-chord point of the extended chord
    due to lift caused by the propeller slipstream at angle-of-attack :cite:`bouquet:2017`.
    cs_prime / cs is assumed to be one and we will neglect the contribution of the fuselage.
    Also, the formula seems to have a discontinuity @ delta_f = 30 degree (not the same sign).
    Figure 11 seems to suggest that the the first line of equation 22 should be <0 as is the
    second line, which is what we will use.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
            values=["cruise", "takeoff", "landing"],
        )

    def setup(self):

        flaps_position = self.options["flaps_position"]
        number_of_points = self.options["number_of_points"]

        if flaps_position == "takeoff":
            self.add_input("data:mission:sizing:takeoff:flap_angle", val=10.0, units="deg")

        elif flaps_position == "landing":
            self.add_input("data:mission:sizing:landing:flap_angle", val=30.0, units="deg")

        self.add_input(
            "delta_Cl",
            val=np.nan,
            shape=number_of_points,
            desc="Increase in the lift coefficient downstream of the propeller",
        )
        self.add_input(
            "delta_Cl_AOA_0",
            val=np.nan,
            shape=number_of_points,
            desc="Increase in the lift coefficient downstream of the propeller for an AOA of 0",
        )

        self.add_output(
            "delta_Cm_alpha",
            val=-0.01,
            shape=number_of_points,
            desc="Increase in pitching moment due to lift caused by the propeller slipstream",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        flaps_position = self.options["flaps_position"]

        if flaps_position == "takeoff":
            delta_f = inputs["data:mission:sizing:takeoff:flap_angle"]

        elif flaps_position == "landing":
            delta_f = inputs["data:mission:sizing:landing:flap_angle"]

        else:
            delta_f = 0

        delta_f = np.clip(delta_f, None, 30.0)

        delta_cl_alpha = inputs["delta_Cl"]
        delta_cl_0 = inputs["delta_Cl_AOA_0"]

        delta_cm_alpha = -0.05 * delta_f / 30.0 * (delta_cl_alpha - delta_cl_0)

        outputs["delta_Cm_alpha"] = delta_cm_alpha

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        flaps_position = self.options["flaps_position"]
        number_of_points = self.options["number_of_points"]

        delta_cl_alpha = inputs["delta_Cl"]
        delta_cl_0 = inputs["delta_Cl_AOA_0"]

        if flaps_position == "takeoff":
            delta_f = inputs["data:mission:sizing:takeoff:flap_angle"]
            partials["delta_Cm_alpha", "data:mission:sizing:takeoff:flap_angle"] = np.where(
                delta_f < 30,
                -0.05 * (delta_cl_alpha - delta_cl_0) / 30.0,
                np.zeros(number_of_points),
            )

        elif flaps_position == "landing":
            delta_f = inputs["data:mission:sizing:landing:flap_angle"]
            partials["delta_Cm_alpha", "data:mission:sizing:landing:flap_angle"] = np.where(
                delta_f < 30,
                -0.05 * (delta_cl_alpha - delta_cl_0) / 30.0,
                np.zeros(number_of_points),
            )

        else:
            delta_f = 0.0

        partials["delta_Cm_alpha", "delta_Cl"] = np.eye(number_of_points) * -0.05 * delta_f / 30.0
        partials["delta_Cm_alpha", "delta_Cl_AOA_0"] = (
            np.eye(number_of_points) * 0.05 * delta_f / 30.0
        )
