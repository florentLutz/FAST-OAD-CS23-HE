# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamExhaustThrust(om.ExplicitComponent):
    """Computation of the thrust produced by the exhaust."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("exhaust_velocity", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("exhaust_mass_flow", units="kg/s", val=np.nan, shape=number_of_points)

        self.add_output("exhaust_thrust", units="N", val=200.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        exhaust_mass_flow = self.smooth_array(inputs["exhaust_mass_flow"])
        exhaust_velocity = self.smooth_array(inputs["exhaust_velocity"])
        true_airspeed = inputs["true_airspeed"]

        # In practice, we should rarely be in conditions where the flow at the exhaust is slower
        # than the true airspeed but just in case
        exhaust_thrust = np.clip(
            exhaust_mass_flow * (exhaust_velocity - true_airspeed),
            np.zeros_like(exhaust_mass_flow),
            None,
        )

        outputs["exhaust_thrust"] = exhaust_thrust

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        exhaust_mass_flow = self.smooth_array(inputs["exhaust_mass_flow"])
        exhaust_velocity = self.smooth_array(inputs["exhaust_velocity"])
        true_airspeed = inputs["true_airspeed"]

        partials_m_dot_8 = exhaust_velocity - true_airspeed
        partials_v8 = exhaust_mass_flow
        partials_v0 = -exhaust_mass_flow

        partials["exhaust_thrust", "exhaust_mass_flow"] = np.diag(
            np.where(
                exhaust_velocity > true_airspeed, partials_m_dot_8, np.zeros_like(exhaust_velocity)
            )
        )
        partials["exhaust_thrust", "exhaust_velocity"] = np.diag(
            np.where(exhaust_velocity > true_airspeed, partials_v8, np.zeros_like(exhaust_velocity))
        )
        partials["exhaust_thrust", "true_airspeed"] = np.diag(
            np.where(exhaust_velocity > true_airspeed, partials_v0, np.zeros_like(exhaust_velocity))
        )

    @staticmethod
    def smooth_array(array):
        """
        In some integration tests, some values of arrays have been seen going to absurd values, this
        is a protection against it.
        """

        median_value = np.median(array)
        # Median value should be during cruise. If max value is too far from median value we cut it
        array_without_absurd_values = np.where(
            array > 5 * median_value, np.zeros_like(array), array
        )
        smoothed_array = np.where(
            array > 5 * median_value,
            np.full_like(array, np.max(array_without_absurd_values)),
            array,
        )

        return smoothed_array
