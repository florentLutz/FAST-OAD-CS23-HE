# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import H2_MOL_PER_KG, O2_KG_PER_MOL, DEFAULT_AIR_CONSUMPTION


class PerformancesPEMFCStackBOPAirConsumption(om.ExplicitComponent):
    """
    Computation of the oxidizer consumption for the PEMFC stack, based on the hydrogen
    consumption per flight point.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "fuel_consumption",
            units="kg/s",
            val=np.nan,
            shape=number_of_points,
        )

        self.add_output(
            "air_consumption",
            units="kg/s",
            val=DEFAULT_AIR_CONSUMPTION,
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="air_consumption",
            wrt="fuel_consumption",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=H2_MOL_PER_KG * O2_KG_PER_MOL / (2.0 * 0.21),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["air_consumption"] = (
            inputs["fuel_consumption"] * H2_MOL_PER_KG * O2_KG_PER_MOL / (2.0 * 0.21)
        )
