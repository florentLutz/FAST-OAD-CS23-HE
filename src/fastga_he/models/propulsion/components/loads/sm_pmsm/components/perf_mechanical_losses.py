# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMechanicalLosses(om.ExplicitComponent):
    """
    Computation of the mechanical losses result from frictions between gas and rotating solid or
    between rotating solid and stationary solid. This is obtained from part II.3.3 in
    :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("air_gap_windage_losses", units="kW", val=np.nan, shape=number_of_points)
        self.add_input("rotor_windage_losses", units="kW", val=np.nan, shape=number_of_points)
        self.add_input("bearing_friction_losses", units="kW", val=np.nan, shape=number_of_points)

        self.add_output(
            "mechanical_power_losses",
            units="kW",
            val=2.1,
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="mechanical_power_losses",
            wrt=["rotor_windage_losses", "bearing_friction_losses"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=2.0,
        )
        self.declare_partials(
            of="mechanical_power_losses",
            wrt="air_gap_windage_losses",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["mechanical_power_losses"] = (
            inputs["air_gap_windage_losses"]
            + 2.0 * inputs["rotor_windage_losses"]
            + 2.0 * inputs["bearing_friction_losses"]
        )
